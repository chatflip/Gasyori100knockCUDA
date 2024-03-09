#include "UtilsCuda.cuh"

cudaStream_t createCudaStream() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  return stream;
}