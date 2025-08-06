#include "sgemm_kernel.cuh"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

typedef void (*MySgemmKernel)(const float *, const float *, float *, float, float, int, int, int);

class RunTest {
  public:
    int m, n, k;
    float *ha, *hb, *hc, *hc_gold;
    float *da, *db, *dc, *dc_gold;
    float alpha, beta;

    dim3 gridSize, blockSize;

    float runtime, throughput, runtime_golden, throughput_golden;

    // int bm, bn, bk, rk;
    // bool enable_double_buffer;

    RunTest(int m, int n, int k, dim3 blockSize) : m(m), n(n), k(k), alpha(1.0f), beta(0.0f), blockSize(blockSize) {
        gridSize = dim3(ceilDiv(n, blockSize.x), ceilDiv(m, blockSize.y));
    }

    void allocateMemory() {
        const size_t size_a = sizeof(float) * m * k;
        const size_t size_b = sizeof(float) * k * n;
        const size_t size_c = sizeof(float) * m * n;

        ha = (float *)malloc(size_a);
        hb = (float *)malloc(size_b);
        hc = (float *)malloc(size_c);
        hc_gold = (float *)malloc(size_c);

        // generate random number
        // srand(2025);
        // for (int i = 0; i < m * k; ++i) {
        //     ha[i] = static_cast<float>(rand() % 10 + 1);
        // }
        // for (int i = 0; i < k * n; ++i) {
        //     hb[i] = static_cast<float>(rand() % 10 + 1);
        // }
        // for (int i = 0; i < m * n; ++i) {
        //     hc[i] = static_cast<float>(rand() % 10 + 1);
        //     hc_gold[i] = hc[i];
        // }

        // srand(2025);
        memset(ha, 1.0, sizeof(float) * m * k);
        memset(hb, 1.0, sizeof(float) * k * n);
        memset(hc, 1.0, sizeof(float) * m * n);
        memset(hc_gold, 1.0, sizeof(float) * m * n);

        // allocate memory on GPU
        cudaMalloc(&da, size_a);
        cudaMalloc(&db, size_b);
        cudaMalloc(&dc, size_c);
        cudaMalloc(&dc_gold, size_c);
        cudaMemcpy(da, ha, size_a, cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, size_b, cudaMemcpyHostToDevice);
        cudaMemcpy(dc, hc, size_c, cudaMemcpyHostToDevice);
        cudaMemcpy(dc_gold, hc_gold, size_c, cudaMemcpyHostToDevice);
    }

    // use cublas as golden
    void runGloden() {
        cublasHandle_t handle;
        cublasCreate(&handle);

        cublasStatus_t status =
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, db, n, da, k, &beta, dc_gold, n);

        cublasDestroy(handle);
        cudaMemcpy(hc_gold, dc_gold, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    }

    void runMyGemm(MySgemmKernel kernel) {
        kernel<<<gridSize, blockSize>>>(da, db, dc, alpha, beta, m, n, k);
        cudaMemcpy(hc, dc, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    }

    bool checkResult() {
        float threshold = 1e-6f;
        for (size_t i = 0; i < m * n; i++) {
            if (std::fabs(hc_gold[i] - hc[i]) > threshold) {
                return false;
            }
        }
        return true;
    }

    ~RunTest() {
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
        cudaFree(dc_gold);
        free(ha);
        free(hb);
        free(hc);
        free(hc_gold);
    }
};

struct Shape {
    int m, n, k;
    dim3 blockDim;
    Shape(int m, int n, int k, dim3 blockDim) : m(m), n(n), k(k), blockDim(blockDim) {}
};

int main(int argc, char **argv) {
    constexpr int bm = 16, bn = 16, bk = 8;
    std::vector<Shape> shapes = {{20, 12, 20, dim3(bm, bn)},          {128, 128, 128, dim3(bm, bn)},
                                 {256, 256, 256, dim3(bm, bn)},       {512, 256, 256, dim3(bm, bn)},
                                 {1024, 1024, 1024, dim3(bm, bn, 1)}, {2048, 768, 2048, dim3(bm, bn, 1)},
                                 {4096, 4096, 4096, dim3(bm, bn, 1)}, {6144, 6144, 6144, dim3(bm, bn, 1)}};

    for (auto &shape : shapes) {
        RunTest test(shape.m, shape.n, shape.k, shape.blockDim);
        test.allocateMemory();
        test.runGloden();

        if (argc > 1) {
            if (std::string(argv[1]) == "v1") {
                test.runMyGemm(sgemm_v1_memory_coalesced);
            } else if (std::string(argv[1]) == "v2") {
                test.runMyGemm(sgemm_v2_shared_memory<bm, bn, bk>);
            } else {
                test.runMyGemm(sgemm_v0_naive);
            }
        } else {
            test.runMyGemm(sgemm_v0_naive);
        }

        printf("shape: %d %d %d\n", shape.m, shape.n, shape.k);
        if (!test.checkResult()) {
            printf("kernel result wrong!\n");
        } else {
            printf("kernel result right!\n");
        }
        printf("[My kernel] runtime: %f ms,        throughput: %f tflops\n", test.runtime, test.throughput / 1.0e3);
        printf("[cublas]    runtime golden: %f ms, throughput golden: %f tflops\n", test.runtime_golden,
               test.throughput_golden / 1.0e3);
    }
}
