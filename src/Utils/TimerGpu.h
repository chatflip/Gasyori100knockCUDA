#pragma once
#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <unordered_map>

class TimerGpu {
 public:
  TimerGpu();
  ~TimerGpu();
  void start(const std::string& name);
  void stop(const std::string& name);
  double elapsedMilliseconds(const std::string& name) const;

 private:
  void checkError(cudaError_t result, const std::string& action) const;

  std::unordered_map<std::string, bool> started, stopped;
  std::unordered_map<std::string, cudaEvent_t> startEvents, stopEvents;
};