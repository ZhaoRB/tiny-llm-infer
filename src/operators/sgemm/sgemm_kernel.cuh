#ifndef SGEMM_KERNEL_CUH
#define SGEMM_KERNEL_CUH

inline __host__ __device__ int ceilDiv(const int total, const int group_size) {
    return (total + group_size - 1) / group_size;
}

// A B C 行主序存储
// 这里 tx 表示行号，ty 表示列号
// gridSize = dim3(ceilDiv(m, blockSize.x), ceilDiv(n, blockSize.y))
// ⭕️ 当前实现有错误
__global__ void sgemm_v0_naive(const float *A, const float *B, float *C,
                               float alpha, float beta, int m, int n, int k) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (tx < m && ty < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[tx * k + i] * B[i * n + ty];
        }
        C[tx * n + ty] = alpha * sum + beta * C[tx * n + ty];
    }
}

// 相较于 naive 的实现，区别是更换 threadIdx 到行列的映射，使其能合并访存
__global__ void sgemm_v1_memory_coalesced(const float *A, const float *B,
                                          float *C, float alpha, float beta,
                                          int m, int n, int k) {
    int y = threadIdx.x + blockDim.x * blockIdx.x;
    int x = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < m && y < n) {
        float sum = 0.0;
        for (int i = 0; i < k; i++) {
            sum = sum + A[x * k + i] * B[i * n + y];
        }
        C[x * n + y] = alpha * sum + beta * C[x * n + y];
    }
}

// 每个线程计算 C 中一个元素的值
// 每个 block 计算 bm * bn = (bm * bk) * (bk * bn) 大小
// 本kernel中，令 bm = bn = bk
template <const int bm = 0, const int bn = 0, const int bk = 0>
__global__ void sgemm_v2_shared_memory(const float *A, const float *B, float *C,
                                       float alpha, float beta, int m, int n,
                                       int k) {
    __shared__ float AS[bm * bk];
    __shared__ float BS[bk * bn];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    int x = threadIdx.x;
    int y = threadIdx.y;
    int col = x + blockIdx.x * bm;
    int row = y + blockIdx.y * bn;

    float sum = 0.0;

    int iterNum = ceilDiv(k, bk);
    for (int iter = 0; iter < iterNum; iter++) {
        // load data from global memory to shared memory
        // 原则：让相邻的 thread，访问 global memory 中连续的内存
        // roundNum 表示，线程块的线程搬运数据要搬运几轮
        int roundNum = ceilDiv(bm * bk, bm * bn);
        int A_global_row_ltop = bm * blockIdx.y;
        int A_global_col_ltop = iter * bk;
        for (int ridx = 0; ridx < roundNum; ridx++) {
            int A_global_row = A_global_row_ltop + (tid + ridx * bm * bn) / bk;
            int A_global_col = A_global_col_ltop + (tid + ridx * bm * bn) % bk;
            if (A_global_row < m && A_global_col < k) {
                AS[tid + ridx * bm * bn] = A[A_global_row * k + A_global_col];
            }
        }
        // load B
        roundNum = ceilDiv(bk * bn, bm * bn);
        int B_global_row_ltop = iter * bk;
        int B_global_col_ltop = bn * blockIdx.x;
        for (int ridx = 0; ridx < roundNum; ridx++) {
            int B_global_row = B_global_row_ltop + (tid + ridx * bm * bn) / bn;
            int B_global_col = B_global_col_ltop + (tid + ridx * bm * bn) % bn;
            if (B_global_row < k && B_global_col < n) {
                BS[tid + ridx * bm * bn] = B[B_global_row * n + B_global_col];
            }
        }
        __syncthreads();

        // calculate result
        // bank conflict ?
        for (int i = 0; i < bk; i++) {
            sum += AS[threadIdx.y * bk + i] * BS[i * bn + threadIdx.x];
        }
        __syncthreads();
    }
    // write back to global memory
    int c_row = blockIdx.y * blockDim.y + threadIdx.y;
    int c_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_row < m && c_col < n) {
        C[c_row * n + c_col] = alpha * sum + beta * C[c_row * n + c_col];
    }
}

#endif // SGEMM_KERNEL_CUH