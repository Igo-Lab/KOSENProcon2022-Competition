#include "add_vector.h"

__global__ void add_vector_dev(float *dest, float *lhs, float *rhs, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        dest[idx] = lhs[idx] + rhs[idx];
    }
}

void add_vector(float *dest, float *lhs, float *rhs, size_t n) {
    float *dev_dest, *dev_lhs, *dev_rhs;
    size_t bytes = n * sizeof(float);

    cudaMalloc((void **)&dev_dest, bytes);
    cudaMalloc((void **)&dev_lhs, bytes);
    cudaMalloc((void **)&dev_rhs, bytes);

    cudaMemcpy(dev_lhs, lhs, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_rhs, rhs, bytes, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((n + block.x - 1) / block.x);

    add_vector_dev<<<grid, block>>>(dev_dest, dev_lhs, dev_rhs, n);

    cudaMemcpy(dest, dev_dest, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_dest);
    cudaFree(dev_lhs);
    cudaFree(dev_rhs);
}
