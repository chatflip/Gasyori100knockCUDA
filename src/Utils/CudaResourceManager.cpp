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
  streams = createCudaStreams(maxStreams);
}

CudaResourceManager::~CudaResourceManager() {
  // Free all the streams
  for (auto stream : streams) {
    cudaStreamDestroy(stream);
  }
}

std::vector<cudaStream_t> CudaResourceManager::createCudaStreams(
    int numStreams) {
  std::vector<cudaStream_t> cudaStreams;
  for (int i = 0; i < numStreams; i++) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStreams.push_back(stream);
  }
  return cudaStreams;
}

std::vector<cudaStream_t> CudaResourceManager::getStreams(int numStreams) {
  std::vector<cudaStream_t> cudaStreams;
  for (int i = 0; i < numStreams; i++) {
    cudaStreams.push_back(streams.at(i));
  }
  return cudaStreams;
}
