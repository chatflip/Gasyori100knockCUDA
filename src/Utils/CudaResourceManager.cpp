#include "CudaResourceManager.hpp"

CudaResourceManager::CudaResourceManager() {
  // Initialize the CUDA context
  cudaError_t cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    std::cerr
        << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"
        << std::endl;
    exit(1);
  }
}

CudaResourceManager::~CudaResourceManager() {
  // Free all the streams
  for (auto stream : streams) {
    cudaStreamDestroy(stream);
  }
}

std::vector<cudaStream_t> CudaResourceManager::getStreams(int numStreams) {
  for (int i = 0; i < numStreams; i++) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams.push_back(stream);
  }
  return streams;
}
