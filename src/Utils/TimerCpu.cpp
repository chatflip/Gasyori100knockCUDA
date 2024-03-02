#include "TimerCpu.h"

TimerCpu::TimerCpu() {
  if (!QueryPerformanceFrequency(&frequency)) {
    std::cerr << "High precision counters are not supported by this system"
              << std::endl;
  }
}

TimerCpu::~TimerCpu() {}

void TimerCpu::start(const std::string& name) {
  started[name] = true;
  stopped[name] = false;
  QueryPerformanceCounter(&startTimes[name]);
}

void TimerCpu::stop(const std::string& name) {
  stopped[name] = true;
  QueryPerformanceCounter(&endTimes[name]);
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
