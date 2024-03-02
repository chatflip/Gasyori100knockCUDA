#pragma once
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>

class TimerGpu {
 public:
  TimerGpu();
  ~TimerGpu();
  void reset();
  void start(const std::string& name);
  void stop(const std::string& name);
  double elapsedMilliseconds(const std::string& name) const;
  void writeToFile(const std::string& path) const;

 private:
  void checkError(cudaError_t result, const std::string& action) const;

  std::map<std::string, bool> started, stopped;
  std::map<std::string, cudaEvent_t> startEvents, stopEvents;
};