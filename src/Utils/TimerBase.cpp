#include "TimerBase.h"

std::string TimerBase::createHeader(const std::string& testName) const {
  std::ostringstream header;
  header << testName << std::endl;

  std::string buildType = "";
#if _DEBUG
  header << "Build: Debug" << std::endl;
#else
  header << "Build: Release" << std::endl;
#endif
  return header.str();
};

std::string TimerBase::createFooter(const float elapsedTime) const {
  std::ostringstream footer;
  footer << std::fixed << std::setprecision(2);
  footer << "Elapsed time: " << elapsedTime << " ms" << std::endl;
  return footer.str();
};

void TimerBase::print(const std::string& header,
                      const std::string& footer) const {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2) << header;
  for (auto& it : started) {
    if (stopped.count(it.first) == 0) {
      std::cerr << "Timer Error: " << it.first << " has not been stopped."
                << std::endl;
      continue;
    }

    if (stopped.at(it.first)) {
      ss << it.first << " " << elapsedMilliseconds(it.first) << " ms"
         << std::endl;
    }
  }
  ss << footer;
  std::cout << ss.str();
}

double TimerBase::calculateTotal(std::vector<std::string> ignoreNames) const {
  double sum = 0.0;

  for (auto& it : started) {
    if (stopped.count(it.first) == 0) {
      std::cerr << "Timer Error: " << it.first << " has not been stopped."
                << std::endl;
      continue;
    }

    if (stopped.at(it.first)) {
      auto isContain =
          std::find(ignoreNames.begin(), ignoreNames.end(), it.first);
      if (isContain != ignoreNames.end()) {
        continue;
      }
      sum += elapsedMilliseconds(it.first);
    }
  }
  return sum;
}

void TimerBase::writeToFile(const std::string& path, const std::string& header,
                            const std::string& footer) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    std::cerr << "Error: could not open file " << path << std::endl;
    return;
  }

  ofs << std::fixed << std::setprecision(2);
  ofs << header;
  for (auto& it : started) {
    if (stopped.count(it.first) == 0) {
      std::cerr << "Timer Error: " << it.first << " has not been stopped."
                << std::endl;
      continue;
    }

    if (stopped.at(it.first)) {
      ofs << it.first << " " << elapsedMilliseconds(it.first) << " ms"
          << std::endl;
    }
  }
  ofs << footer;
  ofs.close();
}