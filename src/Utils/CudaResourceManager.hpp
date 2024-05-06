#pragma once
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <iostream>
#include <vector>

class CudaResourceManager {
 public:
  CudaResourceManager();
  ~CudaResourceManager();
  std::vector<cudaStream_t> getStreams(int numStreams = 1);
  void pushMarker(std::string markerName);
  void popMarker();

 private:
  const int maxStreams = 64;
  std::vector<cudaStream_t> createCudaStreams(int numStreams);
  std::vector<cudaStream_t> streams;
  bool isMarkered = false;
};