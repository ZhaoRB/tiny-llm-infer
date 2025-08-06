#ifndef UTILS
#define UTILS

inline __host__ __device__ int ceilDiv(const int total, const int group_size) {
    return (total + group_size - 1) / group_size;
}

#endif
