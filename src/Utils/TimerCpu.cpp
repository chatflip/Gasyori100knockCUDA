#include "TimerCpu.h"

TimerCpu::TimerCpu() {
  if (!QueryPerformanceFrequency(&frequency)) {
    std::cerr << "High precision counters are not supported by this system"
              << std::endl;
  }
}

TimerCpu::~TimerCpu() {}

void TimerCpu::reset() {
  started.clear();
  stopped.clear();
  startTimes.clear();
  endTimes.clear();
}

void TimerCpu::start(const std::string& name) {
  started[name] = true;
  stopped[name] = false;
  QueryPerformanceCounter(&startTimes[name]);
}

void TimerCpu::stop(const std::string& name) {
  stopped[name] = true;
  QueryPerformanceCounter(&endTimes[name]);
}

double TimerCpu::calculateTotal(std::vector<std::string> ignoreNames) const {
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

double TimerCpu::elapsedMilliseconds(const std::string& name) const {
  if (started.count(name) == 0) {
    std::cerr << "Timer Error: " << name << " has not been started."
              << std::endl;
    return -1.0;
  }

  if (stopped.count(name) == 0) {
    std::cerr << "Timer Error: " << name << " has not been stopped."
              << std::endl;
    return -1.0;
  }

  if (!started.at(name) || !stopped.at(name)) {
    std::cerr << "Timer Error: " << name
              << "Start and stop must be called before getting elapsed time."
              << std::endl;
    return 0.0f;
  }

  double interval = static_cast<double>(endTimes.at(name).QuadPart -
                                        startTimes.at(name).QuadPart) /
                    static_cast<double>(frequency.QuadPart);
  return 1e3 * interval;
}

void TimerCpu::print(const std::string& header,
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

void TimerCpu::writeToFile(const std::string& path, const std::string& header,
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