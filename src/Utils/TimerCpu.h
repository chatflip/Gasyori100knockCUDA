#pragma once
#include <windows.h>

#include "TimerBase.h"

class TimerCpu : public TimerBase {
 public:
  TimerCpu();
  ~TimerCpu();
  void start(const std::string& name) override;
  void stop(const std::string& name) override;
  void reset() override;
  void recordAll() override;

 private:
  LARGE_INTEGER frequency;
  std::unordered_map<std::string, LARGE_INTEGER> startTimes, endTimes;
};