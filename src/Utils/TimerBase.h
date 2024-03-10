#pragma once

#pragma comment(lib, "wbemuuid.lib")
#include <Wbemidl.h>
#include <comdef.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

class TimerBase {
 public:
  virtual void start(const std::string& name) = 0;
  virtual void stop(const std::string& name) = 0;
  virtual void reset() = 0;
  virtual void recordAll() = 0;

  std::string createHeader(const std::string& testName) const;
  std::string createFooter(const float elapsedTime) const;
  float getRecord(const std::string& name) const;
  void popRecord(const std::string& name);
  void mergeRecords(TimerBase& other);

  void writeToFile(const std::string& path, const std::string& header = "",
                   const std::string& footer = "") const;
  void print(const std::string& header = "",
             const std::string& footer = "") const;

 protected:
  std::unordered_map<std::string, bool> started, stopped;
  std::unordered_map<std::string, float> records;

 private:
  std::string getWMIInfo(const std::string& wmiClass,
                         const std::string& property) const;
};