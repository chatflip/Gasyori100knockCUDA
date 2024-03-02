#pragma once
#include <windows.h>

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

class TimerCpu {
 public:
  TimerCpu();
  ~TimerCpu();
  void start(const std::string& name);
  void stop(const std::string& name);
  double elapsedMilliseconds(const std::string& name) const;

 private:
  LARGE_INTEGER frequency;
  std::unordered_map<std::string, bool> started, stopped;
  std::unordered_map<std::string, LARGE_INTEGER> startTimes, endTimes;
};