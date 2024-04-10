#include <cuda_runtime.h>

#include <vector>

std::vector<cudaStream_t> createCudaStreams(int numStreams);
