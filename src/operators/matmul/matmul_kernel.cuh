#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H

#include <cuda_runtime.h>

inline __device__ int CEILDIV(int a, int b) { return (a + b - 1) / b; }
inline __device__ float4 FLOAT4_CAST(float a) { return reinterpret_cast<float4 *>(&a)[0]; }

__global__ void matmul_naive(const float *A, const float *B, float *C, int m, int n, int k) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (y < m && x < n) {
        float sum = 0.0;
        for (int kk = 0; kk < k; kk++) {
            sum += A[y * k + kk] * B[kk * n + x];
        }
        C[y * n + x] = sum;
    }
}

// cuda core: vectorized memory access + block tiling + thread tiling + solve bank conflict
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8,
          const int OFFSET = 0>
__global__ void matmul_128x128_8x8_splitK_bcf(const float *A, const float *B, float *C, int m, int n, int k) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;

    // A block load 到 smem 的时候转置，否则 smem -> register 的时候访存不连续
    __shared__ float smem_a[BK][BM + OFFSET];
    __shared__ float smem_b[BK][BN + OFFSET];

    float reg_load_a[4]; // float4, for transpose
    float reg_load_b[4];
    float reg_compute_a[TM];
    float reg_compute_b[TN];
    float reg_c[TM][TN] = {0.0};

    // shared memory index
    // 256 threads, each loads 4 floats (1 float4)
    // A: load BM*BK = 128*8 = 1024 floats, need 256 threads
    // B: load BK*BN = 8*128 = 1024 floats, need 256 threads
    int smem_a_m = tid / (BK / 4);       // BK/4 = 2, so tid/2, range [0, 127]
    int smem_a_k = (tid % (BK / 4)) * 4; // range [0, 4], step 4
    int smem_b_k = tid / (BN / 4);       // BN/4 = 32, so tid/32, range [0, 7]
    int smem_b_n = (tid % (BN / 4)) * 4; // range [0, 124], step 4

    for (int bk = 0; bk < CEILDIV(k, BK); bk++) {
        // gmem -> smem
        {
            // global memory index
            int gmem_a_m = by * BM + smem_a_m;
            int gmem_a_k = bk * BK + smem_a_k;
            int gmem_a_idx = gmem_a_m * k + gmem_a_k;
            int gmem_b_k = bk * BK;
            int gmem_b_n = bx * BN + smem_b_n;
            int gmem_b_idx = gmem_b_k * n + gmem_b_n;

            // global memory to register
            FLOAT4_CAST(reg_load_a[0]) = FLOAT4_CAST(A[gmem_a_idx]);
            FLOAT4_CAST(reg_load_b[0]) = FLOAT4_CAST(B[gmem_b_idx]);

            // register to shared memory (transpose for A)
            smem_a[smem_a_k][smem_a_m] = reg_load_a[0];
            smem_a[smem_a_k + 1][smem_a_m] = reg_load_a[1];
            smem_a[smem_a_k + 2][smem_a_m] = reg_load_a[2];
            smem_a[smem_a_k + 3][smem_a_m] = reg_load_a[3];
            FLOAT4_CAST(smem_b[smem_b_k][smem_b_n]) = FLOAT4_CAST(reg_load_b[0]);

            __syncthreads();
        }

        // smem -> register & compute
        {
            for (int tk = 0; tk < BK; tk++) {
                FLOAT4_CAST(reg_compute_a[0]) = FLOAT4_CAST(smem_a[tk][ty * TM]);
                FLOAT4_CAST(reg_compute_a[4]) = FLOAT4_CAST(smem_a[tk][ty * TM + 4]);
                FLOAT4_CAST(reg_compute_b[0]) = FLOAT4_CAST(smem_b[tk][tx * TN]);
                FLOAT4_CAST(reg_compute_b[4]) = FLOAT4_CAST(smem_b[tk][tx * TN + 4]);

                // compute
                for (int tm = 0; tm < TM; tm++) {
                    for (int tn = 0; tn < TN; tn++) {
                        reg_c[tm][tn] += reg_compute_a[tm] * reg_compute_b[tn];
                    }
                }
            }
            __syncthreads();
        }
    }

    // write back
    // todo: 可以合并访存写回
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            int gmem_c_m = by * BM + ty * TM + tm;
            int gmem_c_n = bx * BN + tx * TN + tn;
            C[gmem_c_m * n + gmem_c_n] = reg_c[tm][tn];
        }
    }
}

// cuda core: vectorized memory access + block tiling + thread tiling + double buffer + solve bank conflict
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8,
          const int OFFSET = 0>
__global__ void matmul_128x128_8x8_splitK_bcf_dbuf(const float *A, const float *B, float *C, int m, int n, int k) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = tx + blockDim.x * ty;

    // A block load 到 smem 的时候转置，否则 smem -> register 的时候访存不连续
    __shared__ float smem_a[2][BK][BM];
    __shared__ float smem_b[2][BK][BN];

    float reg_load_a[4]; // float4, for transpose
    float reg_load_b[4];
    float reg_compute_a[TM];
    float reg_compute_b[TN];
    float reg_c[TM][TN];

    // shared memory index
    int smem_a_m = tid / 2;
    int smem_a_k = (tid & 1) << 2; // 相当于: (tid % 2) * 4; 0,4
    int smem_b_k = tid / 32;
    int smem_b_n = (tid & 31) << 2; // 相当于: (tid % 32) * 4; 0,4,8,12,...,28

    // load first block to shared memory
    {
        int gmem_a_idx = smem_a_m * k + smem_a_k;
        int gmem_b_idx = smem_b_k * n + smem_b_n;
        FLOAT4_CAST(reg_load_a[0]) = FLOAT4_CAST(A[gmem_a_idx]);
        FLOAT4_CAST(reg_load_b[0]) = FLOAT4_CAST(B[gmem_b_idx]);
    }

    for (int bk = 1; bk < CEILDIV(k, BK); bk++) {
        // global memory index
        int gmem_a_m = by * blockDim.y + smem_a_m;
        int gmem_a_k = bk * BK + smem_a_k;
        int gmem_a_idx = gmem_a_m * k + gmem_a_k;
        int gmem_b_k = bk * BK;
        int gmem_b_n = bx * blockDim.x + smem_b_n;
        int gmem_b_idx = gmem_b_k * n + gmem_b_n;

        // global memory to register
        FLOAT4_CAST(reg_load_a[0]) = FLOAT4_CAST(A[gmem_a_idx]);
        FLOAT4_CAST(reg_load_b[0]) = FLOAT4_CAST(B[gmem_b_idx]);

        // register to shared memory
        smem_a[smem_a_k][smem_a_m] = reg_load_a[0];
        smem_a[smem_a_k][smem_a_m + 1] = reg_load_a[1];
        smem_a[smem_a_k][smem_a_m + 2] = reg_load_a[2];
        smem_a[smem_a_k][smem_a_m + 3] = reg_load_a[3];
        FLOAT4_CAST(smem_b[smem_b_k][smem_b_n]) = FLOAT4_CAST(reg_load_b[0]);

        for (int tm = 0; tm < BM / TM; tm++) {
            for (int tn = 0; tn < BN / TN; tn++) {
                for (int tk = 0; tk < BK; tk++) { // todo: 在最外层 loop tk，顺存访问 shared memory
                    FLOAT4_CAST(reg_compute_a[0]) = FLOAT4_CAST(smem_a[tk][tm * TM]);
                    FLOAT4_CAST(reg_compute_a[4]) = FLOAT4_CAST(smem_a[tk][tm * TM + 4]);
                    FLOAT4_CAST(reg_compute_b[0]) = FLOAT4_CAST(smem_b[tk][tn * TN]);
                    FLOAT4_CAST(reg_compute_b[4]) = FLOAT4_CAST(smem_b[tk][tn * TN + 4]);

                    // compute
                }
            }
        }
    }
}

#endif