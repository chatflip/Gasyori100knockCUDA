#include "TimerCpu.h"

TimerCpu::TimerCpu() : started(false), stopped(false) {
  if (!QueryPerformanceFrequency(&frequency)) {
    std::cerr << "High precision counters are not supported by this system"
              << std::endl;
  }
}

void TimerCpu::start() {
  started = true;
  stopped = false;
  QueryPerformanceCounter(&startTime);
}

void TimerCpu::stop() {
  stopped = true;
  QueryPerformanceCounter(&endTime);
}

double TimerCpu::elapsedMilliseconds() const {
  if (!started || !stopped) {
    std::cerr << "Timer error: Start and stop must be called before getting "
                 "elapsed time."
              << std::endl;
    return 0.0f;
  }

  double interval = static_cast<double>(endTime.QuadPart - startTime.QuadPart) /
                    static_cast<double>(frequency.QuadPart);
  return 1e3 * interval;
}
