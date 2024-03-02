#pragma once
#include <cuda_runtime.h>

#include <iostream>

class TimerGpu {
 public:
  TimerGpu();
  ~TimerGpu();
  void start();
  void stop();
  void reset();
  double elapsedMilliseconds() const;

 private:
  cudaEvent_t startEvent, stopEvent;
  bool started, stopped;

   void checkError(cudaError_t result, const char *action) const;
};