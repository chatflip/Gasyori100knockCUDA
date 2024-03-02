#include "TimerGpu.h"

TimerGpu::TimerGpu() : started(false), stopped(false) {
  checkError(cudaEventCreate(&startEvent), "creating start event");
  checkError(cudaEventCreate(&stopEvent), "creating stop event");
}

TimerGpu::~TimerGpu() {
  checkError(cudaEventDestroy(startEvent), "destroying start event");
  checkError(cudaEventDestroy(stopEvent), "destroying stop event");
}

void TimerGpu::reset() {}
void TimerGpu::start() {
  started = true;
  stopped = false;
  checkError(cudaEventRecord(startEvent, 0), "recording start event");
}

void TimerGpu::stop() {
  stopped = true;
  checkError(cudaEventRecord(stopEvent, 0), "recording stop event");
  checkError(cudaEventSynchronize(stopEvent), "synchronizing on stop event");
}

double TimerGpu::elapsedMilliseconds() const {
  if (!started || !stopped) {
    std::cerr << "Timer error: Start and stop must be called before getting "
                 "elapsed time."
              << std::endl;
    return 0.0f;
  }
  float milliseconds = 0.0f;
  checkError(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent),
             "calculating elapsed time");
  return milliseconds;
}
