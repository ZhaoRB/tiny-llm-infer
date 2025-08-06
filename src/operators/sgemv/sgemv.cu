#include <cuda_runtime.h>

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void sgemv_kernel(float* input, float* weight, float* output, int m, int n) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    float4* input_float4 = (float4*)input;
    float4* weight_float4 = (float4*)weight + THREAD_PER_BLOCK * blockIdx.x;
}