#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_math.h>
#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE = 1024>
__device__ float block_reduce_sum(float *smem) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_NUM = BLOCK_SIZE / WARP_SIZE;

    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE; // 是不是等于 tid & (WARP_SIZE - 1)

    float val = smem[tid];
    val = warp_reduce_sum(val);

    __shared__ float smem_final_reduce[WARP_SIZE];
    if (laneId == 0) {
        smem_final_reduce[warpId] = val;
    }
    __syncthreads();

    val = laneId < WARP_NUM ? smem_final_reduce[laneId] : 0.0;
    val += warp_reduce_sum(val);

    return val;
}

// 第一阶段：计算总的平方和
template <int BLOCK_SIZE>
__global__ void rmsnorm_sum_kernel(const float *input, float *sum, const int vec_len) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float smem_reduce_sum[BLOCK_SIZE];
    float4 input4 = *reinterpret_cast<const float4 *>(input + (bid * BLOCK_SIZE + tid) * 4);
    smem_reduce_sum[tid] = input4.x * input4.x + input4.y * input4.y + input4.z * input4.z + input4.w * input4.w;
    __syncthreads();

    float block_sum = block_reduce_sum<BLOCK_SIZE>(smem_reduce_sum);
    if (tid == 0) {
        atomicAdd(sum, block_sum);
    }
}

// 第二阶段：使用计算好的RMS进行归一化
template <int BLOCK_SIZE>
__global__ void rmsnorm_norm_kernel(const float *input, const float *weight, float *output, const float rms_value,
                                    const int vec_len) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float4 input4 = *reinterpret_cast<const float4 *>(input + (bid * BLOCK_SIZE + tid) * 4);
    float4 weight4 = *reinterpret_cast<const float4 *>(weight + (bid * BLOCK_SIZE + tid) * 4);

    float inv_rms = 1.0f / rms_value;
    float4 output4 = {input4.x * inv_rms * weight4.x, input4.y * inv_rms * weight4.y, input4.z * inv_rms * weight4.z,
                      input4.w * inv_rms * weight4.w};
    *reinterpret_cast<float4 *>(output + (bid * BLOCK_SIZE + tid) * 4) = output4;
}

// 原来的单kernel实现（已修复，但推荐使用两阶段方法）
template <int BLOCK_SIZE>
__global__ void rmsnorm_kernel_float4(const float *input, const float *weight, float *output, float *sum,
                                      const int vec_len, const float eps) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float smem_reduce_sum[BLOCK_SIZE];
    float4 input4 = *reinterpret_cast<const float4 *>(input + (bid * BLOCK_SIZE + tid) * 4);
    smem_reduce_sum[tid] = input4.x * input4.x + input4.y * input4.y + input4.z * input4.z + input4.w * input4.w;
    __syncthreads();

    float block_sum = block_reduce_sum<BLOCK_SIZE>(smem_reduce_sum);
    if (tid == 0) {
        atomicAdd(sum, block_sum);
    }

    // 计算RMS并归一化
    float rms = sqrtf(*sum / vec_len + eps);
    float inv_rms = 1.0f / rms;

    float4 weight4 = *reinterpret_cast<const float4 *>(weight + (bid * BLOCK_SIZE + tid) * 4);
    float4 output4 = {input4.x * inv_rms * weight4.x, input4.y * inv_rms * weight4.y, input4.z * inv_rms * weight4.z,
                      input4.w * inv_rms * weight4.w};
    *reinterpret_cast<float4 *>(output + (bid * BLOCK_SIZE + tid) * 4) = output4;
}