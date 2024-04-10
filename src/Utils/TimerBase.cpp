#include "TimerBase.hpp"

std::string TimerBase::createHeader(const std::string& testName) const {
  std::ostringstream header;
  header << testName << std::endl;
  auto cpus = hwinfo::getAllCPUs();
  auto gpus = hwinfo::getAllGPUs();
  header << "CPU Name: " << cpus.at(0).modelName() << std::endl;
  header << "GPU Name: " << gpus.at(0).name() << std::endl;

#if _DEBUG
  header << "Build: Debug" << std::endl;
#else
  header << "Build: Release" << std::endl;
#endif
  header << "-----------------------------------------\n";
  return header.str();
};

std::string TimerBase::createFooter(const float elapsedTime) const {
  std::ostringstream footer;
  footer << "-----------------------------------------\n";
  footer << std::fixed << std::setprecision(2);
  footer << "Elapsed time: " << elapsedTime << " ms" << std::endl;
  return footer.str();
};

void TimerBase::print(const std::string& header,
                      const std::string& footer) const {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2) << header;
  for (auto& it : records) {
    ss << it.first << " " << it.second << " ms" << std::endl;
  }
  ss << footer;
  std::cout << ss.str();
}

void TimerBase::writeToFile(const std::string& path, const std::string& header,
                            const std::string& footer) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    std::cerr << "Error: could not open file " << path << std::endl;
    return;
  }
  ofs << std::fixed << std::setprecision(2) << header;
  for (auto& it : records) {
    ofs << it.first << " " << it.second << " ms" << std::endl;
  }
  ofs << footer;
  ofs.close();
}

float TimerBase::getRecord(const std::string& name) const {
  if (records.count(name) == 0) {
    std::cerr << "Timer Error: " << name << " has not been recorded."
              << std::endl;
    return -1.0f;
  }
  return records.at(name);
}

void TimerBase::popRecord(const std::string& name) {
  if (records.count(name) == 0) {
    std::cerr << "Timer Error: " << name << " has not been recorded."
              << std::endl;
    return;
  }
  records.erase(name);
}

void TimerBase::mergeRecords(TimerBase& other) {
  for (auto& it : other.records) {
    records[it.first] = it.second;
  }
  other.records.clear();
}
