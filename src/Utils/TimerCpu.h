#pragma once
#include <windows.h>

#include <chrono>
#include <iostream>

class TimerCpu {
 public:
  TimerCpu();
  ~TimerCpu() = default;
  void start();
  void stop();
  void reset();
  double elapsedMilliseconds() const;

 private:
  LARGE_INTEGER frequency;
  LARGE_INTEGER startTime;
  LARGE_INTEGER endTime;
};