#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

inline unsigned int ceildiv(int a, int b) { return (a + b - 1) / b; }

// naive 实现
// 每个线程做一个数据的transpose
// 如果每个线程块的大小是 32*8，有8个warp
// 对于 in 的读取，可以实现合并访存
// 对于 out 的写回，完全没有合并访存
// 如果线程块的大小是 16 * 16
// 对于 out 的写回，每个 warp 能有两个线程合并访存
__global__ void transpose_v1(float *in, float *out, int m, int n) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y < m && x < n) {
        int in_idx = y * n + x;
        int out_idx = x * m + y;
        out[out_idx] = in[in_idx];
    }
}

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>((&pointer))[0])

// 每个线程在寄存器内做 4*4 个块的转置，然后统一写回
__global__ void transpose_v2(float *in, float *out, int m, int n) {
    float in_reg[4][4];
    float out_reg[4][4];

    int col = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    int row = (blockDim.y * blockIdx.y + threadIdx.y) * 4;

    if (row + 3 < m && col + 3 < n) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            FETCH_FLOAT4(in_reg[i]) = FETCH_FLOAT4(in[(row + i) * n + col]);
        }

#pragma unroll
        for (int i = 0; i < 4; i++) {
            FETCH_FLOAT4(out_reg[i]) = {in_reg[0][i], in_reg[1][i],
                                        in_reg[2][i], in_reg[3][i]};
        }

#pragma unroll
        for (int i = 0; i < 4; i++) {
            FETCH_FLOAT4(out[(col + i) * m + row]) = FETCH_FLOAT4(out_reg[i]);
        }
    }
}

void compare(float *in, float *kernel_out, int m, int n) {
    float *gold_out = (float *)malloc(sizeof(float) * m * n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            gold_out[j * m + i] = in[i * n + j];
        }
    }
    for (int i = 0; i < m * n; i++) {
        if (std::abs(kernel_out[i] - gold_out[i]) > 1e-5) {
            std::cout << "kernel Result Wrong!" << std::endl;
            free(gold_out);
            return;
        }
    }
    std::cout << "kernel Result Right!" << std::endl;
    free(gold_out);
}

typedef void (*TransposeKernel)(float *in, float *out, int m, int n);

void test(TransposeKernel kernel, int m, int n, dim3 blockDim, dim3 gridDim) {
    float *h_in, *h_out;
    h_in = (float *)malloc(sizeof(float) * m * n);
    h_out = (float *)malloc(sizeof(float) * m * n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            h_in[i * n + j] = j % 2 == 0 ? 1 : 2;
        }
    }

    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, sizeof(float) * m * n);
    cudaMalloc((void **)&d_out, sizeof(float) * m * n);
    cudaMemcpy(d_in, h_in, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    kernel<<<gridDim, blockDim>>>(d_in, d_out, m, n);

    cudaMemcpy(h_out, d_out, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    compare(h_in, h_out, m, n);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}

int main() {
    const int m = 2048;
    const int n = 4096;

    std::vector<dim3> blockDims = {{32, 8, 1}, {16, 16, 1}, {8, 32, 1}};

    for (auto blockDim : blockDims) {
        dim3 gridDim = {ceildiv(n, blockDim.x), ceildiv(m, blockDim.y), 1};
        test(transpose_v1, m, n, blockDim, gridDim);
    }

    for (auto blockDim : blockDims) {
        dim3 gridDim = {ceildiv(n, blockDim.x * 4), ceildiv(m, blockDim.y * 4),
                        1};
        test(transpose_v2, m, n, blockDim, gridDim);
    }
}
