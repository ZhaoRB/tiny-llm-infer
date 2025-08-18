#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>

// 使用xor，使warp中的每个线程都得到reduce sum的结果
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 因为block size最大是1024，所以最多有32个warp（32*32=1024）
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_NUM = BLOCK_SIZE / WARP_SIZE;
    __shared__ float shared[WARP_NUM];

    int warpid = threadIdx.x / WARP_SIZE;
    int laneid = threadIdx.x % WARP_SIZE;

    val = warp_reduce_sum(val);
    if (laneid == 0) {
        shared[warpid] = val;
    }
    __syncthreads();

    val = (laneid < WARP_NUM) ? shared[laneid] : 0.0f;
    val = warp_reduce_sum(val);
    return val;
}

// ---------------------------softmax fp32-------------------------------------
// 使用一个block执行softmax
// 没有做边界条件处理，size 必须能整除 block_size
template <int BLOCK_SIZE, int SIZE>
__global__ void softmax_kernel_v1(float *in, float *out, float *total, const int size) {
    int tid = threadIdx.x;

    __shared__ float s_in[SIZE];
    __shared__ float s_reduce[BLOCK_SIZE];

    constexpr int times = SIZE / BLOCK_SIZE;
    s_in[tid] = expf(in[tid]);
    s_reduce[tid] = s_in[tid];

    for (int i = 1; i < times; i++) {
        s_in[i * BLOCK_SIZE + tid] = expf(in[i * BLOCK_SIZE + tid]);
        s_reduce[tid] += s_in[i * BLOCK_SIZE + tid];
    }
    __syncthreads();

    float sum = s_reduce[tid];
    sum = block_reduce_sum<BLOCK_SIZE>(sum);

    for (int i = 0; i < times; i++) {
        out[i * BLOCK_SIZE + tid] = s_in[i * BLOCK_SIZE + tid] / sum;
    }
}

// 一个block计算 BLOCK_SIZE * 4 = size个元素
template <int BLOCK_SIZE>
__global__ void softmax_kernel_float4(float *in, float *out, float *total, const int size) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * size + tid;
    float4 in4 = *((float4 *)in + idx);
    float4 in4_exp;
    in4_exp.x = expf(in4.x);
    in4_exp.y = expf(in4.y);
    in4_exp.z = expf(in4.z);
    in4_exp.w = expf(in4.w);

    float sum = in4_exp.x + in4_exp.y + in4_exp.z + in4_exp.w;
    sum = block_reduce_sum<BLOCK_SIZE>(sum);
    if (tid == 0) {
        atomicAdd(total, sum);
    }

    float4 *out4 = (float4 *)out + idx;
    out4->x = in4_exp.x / (*total);
    out4->y = in4_exp.y / (*total);
    out4->z = in4_exp.z / (*total);
    out4->w = in4_exp.w / (*total);
}

template <int BLOCK_SIZE>
__global__ void softmax_kernel_v3(float *in, float *out, float *total, float *finished_blocks, const int size) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * BLOCK_SIZE + tid;
    float4 in4 = *((float4 *)in + idx);
    float4 in4_exp;
    in4_exp.x = expf(in4.x);
    in4_exp.y = expf(in4.y);
    in4_exp.z = expf(in4.z);
    in4_exp.w = expf(in4.w);

    float sum = in4_exp.x + in4_exp.y + in4_exp.z + in4_exp.w;
    sum = block_reduce_sum<BLOCK_SIZE>(sum);

    if (tid == 0) {
        atomicAdd(total, sum);
        atomicAdd(finished_blocks, 1);
        while (atomicAdd(finished_blocks, 0) < gridDim.x) {
            // 自旋等待
        }
    }
    // 上面每个block只让一个线程进行自旋等待，所以需要在这里 __syncthreads()
    // 如果让所有的线程都自旋等待，都执行 atomicAdd，那都会串行化，性能会很差
    __syncthreads();

    float4 *out4 = (float4 *)out + idx;
    out4->x = in4_exp.x / (*total);
    out4->y = in4_exp.y / (*total);
    out4->z = in4_exp.z / (*total);
    out4->w = in4_exp.w / (*total);
}

// ---------------------------test-------------------------------------

typedef void (*SoftmaxKernel_t1)(float *in, float *out, float *total, int size);
typedef void (*SoftmaxKernel_t2)(float *in, float *out, float *total, float *finished_blocks, int size);

void cpu_softmax(const float *in, float *out, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        out[i] = std::exp(in[i]); // e^x
        sum += out[i];
    }

    for (int i = 0; i < size; ++i) {
        out[i] /= sum;
    }
}

template <int GRID_SIZE, int BLOCK_SIZE, typename KERNEL_FP_TYPE>
void test(KERNEL_FP_TYPE kernel, const int size) {
    float *h_data = new float[size];
    float *h_out = new float[size];

    // 初始化数据
    for (int i = 0; i < size; i++) {
        h_data[i] = 1.0f;
    }

    float *d_in, *d_out, *d_total, *d_finished_blocks;
    cudaMalloc(&d_in, sizeof(float) * size);
    cudaMalloc(&d_out, sizeof(float) * size);
    cudaMalloc(&d_total, sizeof(float));
    cudaMalloc(&d_finished_blocks, sizeof(float));

    cudaMemcpy(d_in, h_data, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemset(d_total, 0, sizeof(float));
    cudaMemset(d_finished_blocks, 0, sizeof(float));

    if constexpr (std::is_same<KERNEL_FP_TYPE, SoftmaxKernel_t1>::value) {
        kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_out, d_total, size);
    } else {
        kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_out, d_finished_blocks, d_total, size);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // ------------------- CPU 计算 --------------------
    float *h_out_cpu = new float[size];
    cpu_softmax(h_data, h_out_cpu, size);
    for (int i = 0; i < size; i += 1000) {
        printf("GPU: %f, CPU: %f\n", h_out[i], h_out_cpu[i]);
    }
    delete[] h_out_cpu;

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;
    delete[] h_out;
}

int main() {
    constexpr int size = 4096;
    constexpr int blocksize1 = 256;
    constexpr int blocksize2 = 512;
    constexpr int blocksize3 = 1024;
    printf("softmax_kernel_v1, blocksize: %d\n", blocksize1);
    test<1, blocksize1, SoftmaxKernel_t1>(softmax_kernel_v1<blocksize1, size>, size);
    printf("softmax_kernel_v1, blocksize: %d\n", blocksize2);
    test<1, blocksize2, SoftmaxKernel_t1>(softmax_kernel_v1<blocksize2, size>, size);
    printf("softmax_kernel_v1, blocksize: %d\n", blocksize3);
    test<1, blocksize3, SoftmaxKernel_t1>(softmax_kernel_v1<blocksize3, size>, size);

    // v2
    printf("softmax_kernel_float4, blocksize: %d, gridsize: %d\n", blocksize3, 1);
    test<1, blocksize3, SoftmaxKernel_t1>(softmax_kernel_float4<blocksize3>, size);

    // v3
    constexpr int gridsize1 = size / 4 / blocksize1;
    constexpr int gridsize2 = size / 4 / blocksize2;
    constexpr int gridsize3 = size / 4 / blocksize3;
    printf("softmax_kernel_v3, blocksize: %d, gridsize: %d\n", blocksize1, gridsize1);
    test<gridsize1, blocksize1, SoftmaxKernel_t2>(softmax_kernel_v3<blocksize1>, size);
    printf("softmax_kernel_v3, blocksize: %d, gridsize: %d\n", blocksize2, gridsize2);
    test<gridsize2, blocksize2, SoftmaxKernel_t2>(softmax_kernel_v3<blocksize2>, size);
    printf("softmax_kernel_v3, blocksize: %d, gridsize: %d\n", blocksize3, gridsize3);
    test<gridsize3, blocksize3, SoftmaxKernel_t2>(softmax_kernel_v3<blocksize3>, size);
    return 0;
}
