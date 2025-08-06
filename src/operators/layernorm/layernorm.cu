#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#define CEIL(x, y) ((x) + (y)-1) / (y);

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
__device__ __forceinline__ float block_reduce_sum_v2(float val) {
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

    if (warpid == 0) {
        val = (laneid < WARP_NUM) ? shared[laneid] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// ---------------------------layernorm-------------------------------------

// 只使用一个 threadblock 计算
template <int BLOCK_SIZE>
__global__ void layernorm_v1(float *in, float *weight, float *bias, float *out,
                             int size, float epsilon) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * BLOCK_SIZE + tid;
    __shared__ float s_in[BLOCK_SIZE * 4];

// todo: 这里能否改进？
#pragma unroll
    for (int i = 0; i < 4; i++) {
        s_in[i * BLOCK_SIZE + tid] = in[i * BLOCK_SIZE + tid];
    }
    __syncthreads();
}

template <int BLOCK_SIZE>
__global__ void layernorm_v2(float *in, float *weight, float *bias, float *out,
                             int size, float epsilon) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * BLOCK_SIZE + tid;
    __shared__ float s_in[BLOCK_SIZE * 4];

// todo: 这里能否改进？
#pragma unroll
    for (int i = 0; i < 4; i++) {
        s_in[i * BLOCK_SIZE + tid] = in[i * BLOCK_SIZE + tid];
    }
    __syncthreads();

    float4 *in4 = reinterpret_cast<float4 *>(in) + idx;
    float sum = in4->x + in4->y + in4->z + in4->w;
    sum = block_reduce_sum_v2<BLOCK_SIZE>(sum) / size;
    float avg = sum / size;
}

typedef void (*LayerNormKernel)(float *in, float *weight, float *bias,
                                float *out, int size, float epsilon);

void cpu_layernorm(const float *in, const float *gamma, const float *beta,
                   float *out, int size, float epsilon) {
    // Step 1: 计算均值
    float mean = 0.0f;
    for (int i = 0; i < size; ++i) {
        mean += in[i];
    }
    mean /= size;

    // Step 2: 计算方差
    float var = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = in[i] - mean;
        var += diff * diff;
    }
    var /= size;

    float rstd = 1.0f / std::sqrt(var + epsilon);

    // Step 3: 标准化并仿射变换
    for (int i = 0; i < size; ++i) {
        float norm = (in[i] - mean) * rstd;
        out[i] = norm * gamma[i] + beta[i];
    }
}

void test(LayerNormKernel layernorm_kernel, const int size) {
    // const int size = 1 << 20; // 1M 个元素
    float *h_data = new float[size];
    float *h_weight = new float[size];
    float *h_bias = new float[size];
    float *h_out = new float[size];
    float eps = 1.0;

    // 初始化数据
    for (int i = 0; i < size; i++) {
        h_data[i] = 1.0f;
        h_weight[i] = 1.0f;
        h_bias[i] = 1.0f;
    }

    const int blockDim = 256;
    const int gridDim = (gridDim + blockDim - 1) / blockDim;

    float *d_in, *d_weight, *d_bias, *d_out;
    cudaMalloc(&d_in, sizeof(float) * size);
    cudaMalloc(&d_weight, sizeof(float) * size);
    cudaMalloc(&d_bias, sizeof(float) * size);
    cudaMalloc(&d_out, sizeof(float) * size);
    cudaMemcpy(d_in, h_data, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, sizeof(float) * size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, sizeof(float) * size, cudaMemcpyHostToDevice);

    layernorm_kernel<<<gridDim, blockDim>>>(d_in, d_weight, d_bias, d_out, size,
                                            eps);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // ------------------- CPU 计算 RMSNorm 的平方和 --------------------
    float *h_out_cpu = new float[size];
    cpu_layernorm(h_data, h_weight, h_bias, h_out_cpu, size, eps);

    for (int i = 0; i < 5; i += 100) {
        printf("GPU: %f, CPU: %f\n", h_out[i], h_out_cpu[i]);
    }

    delete[] h_out_cpu;

    cudaFree(d_in);
    cudaFree(d_weight);
    cudaFree(d_out);
    cudaFree(d_bias);
    delete[] h_data;
    delete[] h_weight;
    delete[] h_out;
    delete[] h_bias;
}

int main() {
    constexpr int size = 4096;
    test(layernorm_v1<size / 4>, size);
    test(layernorm_v2<size / 4>, size);
    return 0;
}
