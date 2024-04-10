#include "UtilsCuda.hpp"

std::vector<cudaStream_t> createCudaStreams(int numStreams) {
  std::vector<cudaStream_t> streams;
  for (int i = 0; i < numStreams; i++) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams.push_back(stream);
  }
  return streams;
}