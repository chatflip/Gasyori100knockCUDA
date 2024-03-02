#pragma once
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "TimerBase.h"

class TimerGpu : public TimerBase {
 public:
  TimerGpu();
  ~TimerGpu();
  void reset();
  void start(const std::string& name) override;
  void stop(const std::string& name) override;
  double elapsedMilliseconds(const std::string& name) const;
  double calculateTotal(std::vector<std::string> ignoreNames) const;
  void writeToFile(const std::string& path, const std::string& header = "",
                   const std::string& footer = "") const;
  void print(const std::string& header = "",
             const std::string& footer = "") const;

 private:
  void checkError(cudaError_t result, const std::string& action) const;

  std::map<std::string, bool> started, stopped;
  std::map<std::string, cudaEvent_t> startEvents, stopEvents;
};