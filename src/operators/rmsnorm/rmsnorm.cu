#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

void cpu_rmsnorm(const float *in, const float *weight, float *out, int size,
                 float epsilon) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum_sq += in[i] * in[i];
    }
    float rms = std::sqrt(sum_sq / size + epsilon);

    for (int i = 0; i < size; ++i) {
        out[i] = (in[i] / rms) * weight[i];
    }
}

__global__ void rmsnorm_kernel(float *in, float *weight, float *out, int size,
                               float epsilon) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int idx = (tid + blockSize * blockIdx.x) * 2;
    extern __shared__ float sdata[];

    // todo: 可能优化？
    if (idx < size) {
        sdata[tid] = in[idx] + in[idx + 1];
    } else if (idx + 1 < size) {
        sdata[tid] = in[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // reduce
    // #pragma unroll
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
            value += __shfl_xor_sync(0xffffffff, value, s, 32);
        }

        // write back
        if (tid == 0) {
            atomicAdd(out, value);
        }
    }
}

int main() {
    const int size = 1 << 20; // 1M 个元素
    // const int size = 1024;
    float *h_data = new float[size];
    float *h_weight = new float[size];
    float *h_out = new float[size];
    float eps = 1.0;

    // 初始化数据
    for (int i = 0; i < size; i++) {
        h_data[i] = 1.0f;
        h_weight[i] = 1.0f;
    }

    const int blockDim = 256;
    const int gridDim = (gridDim + blockDim - 1) / blockDim;

    float *d_in, *d_weight, *d_out;
    cudaMalloc(&d_in, sizeof(float) * size);
    cudaMalloc(&d_weight, sizeof(float) * size);
    cudaMalloc(&d_out, sizeof(float) * size);
    cudaMemcpy(d_in, h_data, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, sizeof(float) * size,
               cudaMemcpyHostToDevice);

    rmsnorm_kernel<<<gridDim, blockDim>>>(d_in, d_weight, d_out, size, eps);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // ------------------- CPU 计算 RMSNorm 的平方和 --------------------
    float *h_out_cpu = new float[size];
    cpu_rmsnorm(h_data, h_weight, h_out_cpu, size, eps);

    for (int i = 0; i < 5; i += 100) {
        printf("GPU: %f, CPU: %f\n", h_out[i], h_out_cpu[i]);
    }

    delete[] h_out_cpu;

    cudaFree(d_in);
    cudaFree(d_weight);
    cudaFree(d_out);
    delete[] h_data;
    delete[] h_weight;
    delete[] h_out;
    return 0;
}
