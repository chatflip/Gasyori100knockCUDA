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
  double elapsedMilliseconds() const;

 private:
  LARGE_INTEGER frequency, startTime, endTime;
  bool started, stopped;
};