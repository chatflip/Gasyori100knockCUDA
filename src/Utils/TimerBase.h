#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class TimerBase {
 public:
  virtual void start(const std::string& name) = 0;
  virtual void stop(const std::string& name) = 0;
  virtual void reset() = 0;
  virtual double elapsedMilliseconds(const std::string& name) const = 0;

  std::string createHeader(const std::string& testName) const;
  std::string createFooter(const float elapsedTime) const;
  double calculateTotal(std::vector<std::string> ignoreNames) const;
  void writeToFile(const std::string& path, const std::string& header = "",
                   const std::string& footer = "") const;
  void print(const std::string& header = "",
             const std::string& footer = "") const;

 protected:
  std::unordered_map<std::string, bool> started, stopped;
};