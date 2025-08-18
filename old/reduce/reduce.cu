#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

// 树形规约
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v1(float *in, float *out, int size) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int idx = tid + blockSize * blockIdx.x;
    __shared__ float sdata[BLOCK_SIZE];

    float2 *in2 = (float2 *)in + idx;
    sdata[tid] = in2->x + in2->y;
    __syncthreads();

// reduce
#pragma unroll
    for (int s = blockSize / 2; s > 16; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // warp reduce using warp shuffle
    if (tid < 32) {
        float value = sdata[tid];
        for (int s = 16; s > 0; s >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, s);
        }
        // write back
        if (tid == 0) {
            atomicAdd(out, value);
        }
    }
}


// ----------------------------------------------------------------
typedef void (*ReduceKernel)(float *in, float *out, int size);

template <size_t blockDim = 256>
float sumArray(const float *h_in, int size, float &avgTime,
               ReduceKernel kernel) {

    float *h_data = new float[size];
    for (int i = 0; i < size; i++) {
        h_data[i] = 1.0f;
    }
    float *d_in, *d_out, h_out;
    const int gridDim = (size + blockDim * 2 - 1) / (blockDim * 2);

    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<gridDim, blockDim>>>(d_in, d_out, size);

    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;

    return h_out;
}

int main() {
    constexpr int size = 1 << 20; // 1M 个元素

    return 0;
}
