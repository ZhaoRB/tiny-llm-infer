#include "matmul_kernel.cuh"
#include <cfloat>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

#define WARMUP_TIME 1
#define TEST_TIME 1

void check_result(const float *hA, const float *hB, float *hC_golden, float *dC, float *hC, int m, int n, int k) {
    printf("Checking matmul results...\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0;
            for (int kk = 0; kk < k; kk++) {
                sum += hA[i * k + kk] * hB[kk * n + j];
            }
            hC_golden[i * n + j] = sum;
        }
    }
    // float alpha = 1.0f, beta = 0.0f;
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, hB, n, hA, k, &beta, hC_golden, n);
    // cublasDestroy(handle);
    printf("Golden result is been caculated, now start checking...\n");

    cudaMemcpy(hC, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    bool is_correct = true;
    float max_error = FLT_MIN;
    float sum_error = 0.0f;

    for (int i = 0; i < m * n; i++) {
        float error = fabsf(hC[i] - hC_golden[i]);
        if (error > 1e-4) {
            printf("Error at index %d: %f != %f, error: %f\n", i, hC[i], hC_golden[i], error);
            is_correct = false;
            if (error > max_error) {
                max_error = error;
            }
            sum_error += error;
        }
    }
    if (is_correct) {
        printf("Test passed\n");
    } else {
        printf("Test failed\n");
        printf("Max error: %f\n", max_error);
        printf("Avg error: %f\n", sum_error / (m * n));
    }
}

class MatmulData {
  public:
    float *hA, *hB, *hC, *hC_golden;
    float *dA, *dB, *dC;
    int m, n, k;

    MatmulData(int m, int n, int k) : m(m), n(n), k(k) {
        hA = (float *)malloc(m * k * sizeof(float));
        hB = (float *)malloc(k * n * sizeof(float));
        hC = (float *)malloc(m * n * sizeof(float));
        hC_golden = (float *)malloc(m * n * sizeof(float));
        cudaMalloc(&dA, m * k * sizeof(float));
        cudaMalloc(&dB, k * n * sizeof(float));
        cudaMalloc(&dC, m * n * sizeof(float));

        std::mt19937 rd(12);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < m * k; i++) {
            hA[i] = dist(rd);
        }
        for (int i = 0; i < k * n; i++) {
            hB[i] = dist(rd);
        }
        for (int i = 0; i < m * n; i++) {
            hC[i] = 0.0;
            hC_golden[i] = 0.0;
        }
        cudaMemcpy(dA, hA, m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, k * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dC, hC, m * n * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~MatmulData() {
        free(hA);
        free(hB);
        free(hC);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
};

void test_matmul_check_result() {
    // std::vector<std::vector<int>> shapes = {
    //     {1024, 1024, 1024}, {1024, 1024, 2048}, {1024, 1024, 4096}, {1024, 1024, 8192}};
    std::vector<std::vector<int>> shapes = {{1024, 1024, 1024}};
    std::vector<dim3> blockDims = {{16, 16, 1}};

    constexpr int BN = 128, BM = 128;
    constexpr int BK = 8;
    constexpr int TN = 8, TM = 8;
    dim3 blockDim = {BN / TN, BM / TM, 1};

    for (auto &shape : shapes) {
        for (auto &blockDim : blockDims) {
            int m = shape[0];
            int n = shape[1];
            int k = shape[2];
            MatmulData matmulData(m, n, k);
            printf("Matmul shape: %d x %d x %d\n", m, n, k);

            dim3 gridDim = {(unsigned int)(n / BN), (unsigned int)(m / BM), 1};
            // matmul_naive<<<gridDim, blockDim>>>(matmulData.dA, matmulData.dB, matmulData.dC, m, n, k);
            matmul_128x128_8x8_splitK_bcf<BM, BN, BK, TM, TN, 0>
                <<<gridDim, blockDim>>>(matmulData.dA, matmulData.dB, matmulData.dC, m, n, k);

            cudaDeviceSynchronize();
            check_result(matmulData.hA, matmulData.hB, matmulData.hC_golden, matmulData.dC, matmulData.hC, m, n, k);
        }
    }
}

struct Shape {
    int m, n, k;
    std::string name;

    Shape(int _m, int _n, int _k, std::string _name) : m(_m), n(_n), k(_k), name(_name) {}
};

void test_matmul_preformance_128x128_8x8_float4() {
    printf("Testing matmul preformance with 128x128_8x8_float4...\n");
    // Qwen3-4B
    std::vector<Shape> shapes = {
        // 4096 = 128 * 32
        // Shape(128, 4096, 2560, "Q_proj_seqLen_hiddenDim_headDimxnumHeads"),
        // Shape(256, 4096, 2560, "Q_proj_seqLen_hiddenDim_headDimxnumHeads"),
        // Shape(512, 4096, 2560, "Q_proj_seqLen_hiddenDim_headDimxnumHeads"),
        // Shape(1024, 4096, 2560, "Q_proj_seqLen_hiddenDim_headDimxnumHeads"),
        // Shape(2048, 4096, 2560, "Q_proj_seqLen_hiddenDim_headDimxnumHeads"),
        // Shape(4096, 4096, 2560, "Q_proj_seqLen_hiddenDim_headDimxnumHeads"),
        // 1024 = 128 * 8
        // Shape(128, 1024, 2560, "KV_proj_seqLen_hiddenDim_headDimxnumHeads"),
        // Shape(256, 1024, 2560, "KV_proj_seqLen_hiddenDim_headDimxnumHeads"),
        // Shape(512, 1024, 2560, "KV_proj_seqLen_hiddenDim_headDimxnumHeads"),
        // Shape(1024, 1024, 2560, "KV_proj_seqLen_hiddenDim_headDimxnumHeads"),
        // Shape(2048, 1024, 2560, "KV_proj_seqLen_hiddenDim_headDimxnumHeads"),
        // Shape(4096, 1024, 2560, "KV_proj_seqLen_hiddenDim_headDimxnumHeads"),
        Shape(1024, 1024, 1024, "custom"),
    };

    constexpr int BN = 128, BM = 128;
    constexpr int BK = 8;
    constexpr int TN = 8, TM = 8;
    dim3 blockDim = {BN / TN, BM / TM, 1};

    for (auto &shape : shapes) {
        int m = shape.m;
        int n = shape.n;
        int k = shape.k;
        MatmulData matmulData(m, n, k);

        dim3 gridDim = {(unsigned int)(n / BN), (unsigned int)(m / BM), 1};
        // warmup
        for (int i = 0; i < WARMUP_TIME; i++) {
            matmul_128x128_8x8_splitK_bcf<BM, BN, BK, TM, TN, 0>
                <<<gridDim, blockDim>>>(matmulData.dA, matmulData.dB, matmulData.dC, m, n, k);
            cudaDeviceSynchronize();
        }

        // create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // test
        float min_duration = std::numeric_limits<float>::max();
        for (int i = 0; i < TEST_TIME; i++) {
            cudaEventRecord(start);
            matmul_128x128_8x8_splitK_bcf<BM, BN, BK, TM, TN, 0>
                <<<gridDim, blockDim>>>(matmulData.dA, matmulData.dB, matmulData.dC, m, n, k);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            if (elapsed_ms < min_duration) {
                min_duration = elapsed_ms;
            }
        }

        // destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("min duration: %f ms\n", min_duration);
        if (min_duration > 0) {
            double FLOPS = 2.0 * m * n * k / (min_duration / 1e3); // convert milliseconds to seconds
            printf("Max performance: %f TFLOPS\n", FLOPS / 1e12);
        } else {
            printf("Error: Unable to measure execution time\n");
        }

        // 使用 cublas 计算并统计时间
        cublasHandle_t handle;
        cublasCreate(&handle);

        // warmup
        float alpha = 1.0f, beta = 0.0f;
        for (int i = 0; i < WARMUP_TIME; i++) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, matmulData.dA, m, matmulData.dB, k, &beta,
                        matmulData.dC, m);
            cudaDeviceSynchronize();
        }

        // test
        cudaEvent_t start_cublas, stop_cublas;
        cudaEventCreate(&start_cublas);
        cudaEventCreate(&stop_cublas);

        float min_duration_cublas = std::numeric_limits<float>::max();
        for (int i = 0; i < TEST_TIME; i++) {
            cudaEventRecord(start_cublas);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, matmulData.dA, m, matmulData.dB, k, &beta,
                        matmulData.dC, m);
            cudaDeviceSynchronize();

            cudaEventRecord(stop_cublas);
            cudaEventSynchronize(stop_cublas);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start_cublas, stop_cublas);
            if (elapsed_ms < min_duration_cublas) {
                min_duration_cublas = elapsed_ms;
            }
        }

        cudaEventDestroy(start_cublas);
        cudaEventDestroy(stop_cublas);

        printf("cublas min duration: %f ms\n", min_duration_cublas);
        if (min_duration_cublas > 0) {
            double FLOPS = 2.0 * m * n * k / (min_duration_cublas / 1e3); // convert milliseconds to seconds
            printf("cublas Max performance: %f TFLOPS\n", FLOPS / 1e12);
        } else {
            printf("Error: Unable to measure cublas execution time\n");
        }

        // destroy cublas handle
        cublasDestroy(handle);
    }
}

int main() {
    // test_matmul_check_result();
    test_matmul_preformance_128x128_8x8_float4();
}
