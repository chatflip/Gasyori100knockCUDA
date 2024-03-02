#pragma once

#include <iomanip>
#include <sstream>
#include <string>

class TimerBase {
 public:
  virtual void start(const std::string& name) = 0;
  virtual void stop(const std::string& name) = 0;
  std::string createHeader(const std::string& testName) const;
  std::string createFooter(const float elapsedTime) const;

 private:
};