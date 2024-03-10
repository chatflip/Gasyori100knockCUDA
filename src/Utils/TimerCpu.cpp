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
  records.clear();
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

void TimerCpu::recordAll() {
  auto it = started.begin();
  while (it != started.end()) {
    std::string name = it->first;
    if (started.count(name) == 0) {
      std::cerr << "Timer Error: " << name << " has not been started."
                << std::endl;
      it++;
      continue;
    }

    if (stopped.count(name) == 0) {
      std::cerr << "Timer Error: " << name << " has not been stopped."
                << std::endl;
      it++;
      continue;
    }

    if (!started.at(name) || !stopped.at(name)) {
      std::cerr << "Timer Error:" << name
                << " Start and stop must be called before getting elapsed time."
                << std::endl;
      it++;
      continue;
    }

    double interval = static_cast<double>(endTimes.at(name).QuadPart -
                                          startTimes.at(name).QuadPart) /
                      static_cast<double>(frequency.QuadPart);
    float milliseconds = 1e3 * interval;
    records[name] = milliseconds;

    it = started.erase(it);
    stopped.erase(name);
    startTimes.erase(name);
    endTimes.erase(name);
  }
}
