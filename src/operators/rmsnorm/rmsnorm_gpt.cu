#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

#define HIDDEN_DIM 4096
#define BLOCK_SIZE 256

// 设备全局变量，用于存储所有线程块的总和
__device__ float global_sum_sq = 0.0f;

// 线程块内归约求和
__device__ float blockReduceSum(float val) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    return shared[0];
}

// 计算 RMSNorm
__global__ void rmsnorm_forward(const float *__restrict__ input,
                                float *__restrict__ output,
                                const float *__restrict__ weight,
                                float epsilon) {
    __shared__ float rms;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float val = (idx < HIDDEN_DIM) ? input[idx] : 0.0f;
    float sq = val * val;

    // 归约每个 block 内的平方和
    float block_sum_sq = blockReduceSum(sq);

    // 使用原子操作累加到全局变量
    if (threadIdx.x == 0) {
        atomicAdd(&global_sum_sq, block_sum_sq);
    }
    __syncthreads();

    // 线程 0 计算 RMS 并广播
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum_sq = global_sum_sq; // 获取所有 block 的平方和
        rms = sqrtf(sum_sq / HIDDEN_DIM + epsilon);
        global_sum_sq = 0.0f; // 复位全局变量（避免影响下次计算）
    }
    __syncthreads();

    // 归一化 & 缩放
    if (idx < HIDDEN_DIM) {
        output[idx] = (val / rms) * weight[idx];
    }
}

// 启动 RMSNorm 计算
void launch_rmsnorm(const float *input, float *output, const float *weight,
                    float epsilon) {
    int numBlocks = (HIDDEN_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rmsnorm_forward<<<numBlocks, BLOCK_SIZE>>>(input, output, weight, epsilon);
}

int main() {
    float epsilon = 1.0f;
    float h_input[HIDDEN_DIM];
    float h_weight[HIDDEN_DIM];
    float h_output[HIDDEN_DIM];

    // 初始化数据
    for (int i = 0; i < HIDDEN_DIM; ++i) {
        h_input[i] = 1.0f;
        h_weight[i] = 1.0f;
    }

    // 申请 GPU 内存
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, sizeof(h_input));
    cudaMalloc(&d_weight, sizeof(h_weight));
    cudaMalloc(&d_output, sizeof(h_output));

    // 拷贝数据到 GPU
    cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, sizeof(h_weight), cudaMemcpyHostToDevice);

    // 启动 RMSNorm 计算
    launch_rmsnorm(d_input, d_output, d_weight, epsilon);

    // 拷贝结果回 CPU
    cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);

    std::cout << "RMSNorm completed!" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "Output[" << i << "]: " << h_output[i] << std::endl;
    }
    return 0;
}
