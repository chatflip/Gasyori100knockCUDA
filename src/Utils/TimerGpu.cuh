#pragma once
#include <cuda_runtime.h>

#include "TimerBase.h"

class TimerGpu : public TimerBase {
 public:
  TimerGpu();
  ~TimerGpu();
  void start(const std::string& name) override;
  void start(const std::string& name, cudaStream_t stream);
  void stop(const std::string& name) override;
  void stop(const std::string& name, cudaStream_t stream,
            bool syncEvent = true);
  void reset() override;
  void recordAll() override;

 private:
  void checkError(cudaError_t result, const std::string& action) const;

  std::unordered_map<std::string, cudaEvent_t> startEvents, stopEvents;
};