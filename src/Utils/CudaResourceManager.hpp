#pragma once
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

class CudaResourceManager {
 public:
  CudaResourceManager();
  ~CudaResourceManager();
  std::vector<cudaStream_t> getStreams(int numStreams = 1);

 private:
  std::vector<cudaStream_t> streams;
};