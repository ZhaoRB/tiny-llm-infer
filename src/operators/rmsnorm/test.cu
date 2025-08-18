#include "rmsnorm.cuh"
#include <__clang_cuda_runtime_wrapper.h>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

template <int BLOCK_SIZE>
class RMSNorm {
  private:
    int size;
    float *h_in, *h_weight, *h_out, *h_out_golden;
    float *d_in, *d_weight, *d_out;
    float eps;

  public:
    RMSNorm(int size) : size(size), eps(1e-5) {
        h_in = new float[size * sizeof(float)];
        h_weight = new float[size * sizeof(float)];
        h_out = new float[size * sizeof(float)]();
        h_out_golden = new float[size * sizeof(float)]();

        std::mt19937 gen(1);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < size; i++) {
            h_in[i] = dis(gen);
            h_weight[i] = dis(gen);
        }

        cudaMalloc(&d_in, size * sizeof(float));
        cudaMalloc(&d_weight, size * sizeof(float));
        cudaMalloc(&d_out, size * sizeof(float));
        cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, h_weight, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    void check_result() {
        printf("Checking result...\n");
        printf("Calculating golden result...\n");
        float sum_sq = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum_sq += h_in[i] * h_in[i];
        }
        float rms = std::sqrt(sum_sq / size + eps);
        for (int i = 0; i < size; ++i) {
            h_out_golden[i] = (h_in[i] / rms) * h_weight[i];
        }
        printf("Golden result calculated\n");

        cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

        printf("Start checking result...\n");
        bool is_correct = true;
        for (int i = 0; i < size; ++i) {
            if (fabsf(h_out[i] - h_out_golden[i]) > 1e-4) {
                printf("Error at index %d: %f != %f\n", i, h_out[i], h_out_golden[i]);
                is_correct = false;
            }
        }
        printf("Result: %s\n", is_correct ? "correct" : "incorrect");
    }

    void forward() {
        dim3 blockDim(BLOCK_SIZE);
        dim3 gridDim((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        rmsnorm_kernel_float4<BLOCK_SIZE><<<gridDim, blockDim>>>(d_in, d_weight, d_out, size, eps);
        cudaDeviceSynchronize();
    }

    void test_performance() {
        constexpr int WARMUP_TIME = 1;
        for (int i = 0; i < WARMUP_TIME; i++) {
            forward();
        }

        constexpr int LOOP_TIME = 10;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < LOOP_TIME; i++) {
            forward();
        }
        cudaEventSynchronize(stop);
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("RMSNorm time: %f ms\n", milliseconds);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    ~RMSNorm() {
        delete[] h_in;
        delete[] h_weight;
        delete[] h_out;
        cudaFree(d_in);
        cudaFree(d_weight);
        cudaFree(d_out);
    }
};

int main() {
    RMSNorm<128> rmsnorm(1024);
    rmsnorm.check_result();
    return 0;
}
