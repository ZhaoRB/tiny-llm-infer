#include <cstdio>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float *input, const float *weight,
                                      float *output, int M, int K) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    int start_row = blockIdx.x * ROW_PER_BLOCK;
    int end_row = start_row + ROW_PER_BLOCK;
    if (start_row >= K) {
        return;
    }

    constexpr int pack_size = 4;
    const int pack_num = M / pack_size;
    const int pack_off = pack_size * pack_num;

#pragma unroll
    for (int p = start_row; p < end_row; ++p) {
        sdata[tid] = 0;
        int row_offset = p * M; //- 计算每一行的起始位置
        float4 *input_float4_ptr = (float4 *)input; //- 转换为 float4 指针
        float4 *weight_float4_ptr = (float4 *)(weight + row_offset);

#pragma unroll
        //- 每个线程一次计算四个数的 乘加
        for (int i = tid; i < pack_num; i += blockDim.x) {
            float4 input_float4 = *(input_float4_ptr + i);
            float4 weight_float4 = *(weight_float4_ptr + i);
            float part_sum = input_float4.x * weight_float4.x +
                             input_float4.y * weight_float4.y +
                             input_float4.z * weight_float4.z +
                             input_float4.w * weight_float4.w;
            sdata[tid] += part_sum;
        }

        //- 处理剩余的
        for (int i = pack_off + tid; i < M; i += blockDim.x) {
            sdata[tid] += input[i] * weight[row_offset + i];
        }

        __syncthreads();

        //- 规约
        using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;
        float part_sum = BlockReduce(temp).Sum(sdata[tid]);
        __syncthreads();

        if (tid == 0) {
            output[p] = part_sum;
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32_1(const float *input, const float *weight,
                                        float *output, int M, int K) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    int start_row = blockIdx.x * ROW_PER_BLOCK;
    int end_row = start_row + ROW_PER_BLOCK;
    if (start_row >= K) {
        return;
    }

#pragma unroll
    for (int p = start_row; p < end_row; ++p) {
        sdata[tid] = 0;
        int row_offset = p * M; //- 计算每一行的起始位置

#pragma unroll
        //- 不使用向量化
        for (int i = tid; i < M; i += blockDim.x) {
            sdata[tid] += input[i] * weight[row_offset + i];
        }

        __syncthreads();

        //- 规约
        using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;
        float part_sum = BlockReduce(temp).Sum(sdata[tid]);
        __syncthreads();

        if (tid == 0) {
            output[p] = part_sum;
        }
        __syncthreads();
    }
}

constexpr int THREAD_PER_BLOCK = 256;
constexpr int ROW_PER_BLOCK = 4;
constexpr int M = 8192;
constexpr int K = 512;

int main() {
    // 1. 分配主机内存
    float *h_input = new float[M];
    float *h_weight = new float[K * M];
    float *h_output = new float[K];

    // 2. 初始化数据（所有元素设为 1）
    std::fill_n(h_input, M, 1.0f);
    std::fill_n(h_weight, K * M, 1.0f);

    // 3. 分配设备内存
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, M * sizeof(float));
    cudaMalloc(&d_weight, K * M * sizeof(float));
    cudaMalloc(&d_output, K * sizeof(float));

    // 4. 拷贝数据到设备
    cudaMemcpy(d_input, h_input, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, K * M * sizeof(float),
               cudaMemcpyHostToDevice);

    //
    // 5. 计算网格和块大小
    int grid_size = (K + ROW_PER_BLOCK - 1) / ROW_PER_BLOCK;
    dim3 grid(grid_size);
    dim3 block(THREAD_PER_BLOCK);

    matmul_kernel_cu_fp32<THREAD_PER_BLOCK, ROW_PER_BLOCK>
        <<<grid, block>>>(d_input, d_weight, d_output, M, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, K * sizeof(float), cudaMemcpyDeviceToHost);


    matmul_kernel_cu_fp32_1<THREAD_PER_BLOCK, ROW_PER_BLOCK>
        <<<grid, block>>>(d_input, d_weight, d_output, M, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, K * sizeof(float), cudaMemcpyDeviceToHost);


    // 9. 释放内存
    delete[] h_input;
    delete[] h_weight;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);

    return 0;
}