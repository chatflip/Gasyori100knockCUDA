#pragma once
#include <windows.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "TimerBase.h"

class TimerCpu : public TimerBase {
 public:
  TimerCpu();
  ~TimerCpu();
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
  LARGE_INTEGER frequency;
  std::map<std::string, bool> started, stopped;
  std::map<std::string, LARGE_INTEGER> startTimes, endTimes;
};