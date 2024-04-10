#include "TimerGpu.hpp"

TimerGpu::TimerGpu() {}

TimerGpu::~TimerGpu() { reset(); }

void TimerGpu::start(const std::string& name) {
  started[name] = true;
  stopped[name] = false;
  cudaEvent_t startEvent;
  checkError(cudaEventCreate(&startEvent), "creating start event " + name);
  checkError(cudaEventRecord(startEvent, 0), "recording start event " + name);
  startEvents[name] = startEvent;
}

void TimerGpu::start(const std::string& name, cudaStream_t stream) {
  started[name] = true;
  stopped[name] = false;
  cudaEvent_t startEvent;
  checkError(cudaEventCreate(&startEvent), "creating start event " + name);
  checkError(cudaEventRecord(startEvent, stream),
             "recording start event " + name);
  startEvents[name] = startEvent;
}

void TimerGpu::stop(const std::string& name) {
  stopped[name] = true;
  cudaEvent_t stopEvent;
  checkError(cudaEventCreate(&stopEvent), "creating stop event " + name);
  checkError(cudaEventRecord(stopEvent, 0), "recording stop event" + name);
  checkError(cudaEventSynchronize(stopEvent),
             "synchronizing on stop event " + name);
  stopEvents[name] = stopEvent;
}

void TimerGpu::stop(const std::string& name, cudaStream_t stream,
                    bool syncEvent) {
  stopped[name] = true;
  cudaEvent_t stopEvent;
  checkError(cudaEventCreate(&stopEvent), "creating stop event " + name);
  checkError(cudaEventRecord(stopEvent, stream), "recording stop event" + name);
  if (syncEvent) {
    checkError(cudaEventSynchronize(stopEvent),
               "synchronizing on stop event " + name);
  }
  stopEvents[name] = stopEvent;
}

void TimerGpu::reset() {
  for (auto& it : startEvents) {
    checkError(cudaEventDestroy(it.second),
               "destroying start event " + it.first);
  }
  for (auto& it : stopEvents) {
    checkError(cudaEventDestroy(it.second), "destroying end event " + it.first);
  }
  started.clear();
  stopped.clear();
  startEvents.clear();
  stopEvents.clear();
  records.clear();
}

void TimerGpu::recordAll() {
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

    float milliseconds = 0.0f;
    checkError(cudaEventElapsedTime(&milliseconds, startEvents.at(name),
                                    stopEvents.at(name)),
               "calculating elapsed time " + name);
    records[name] = milliseconds;

    it = started.erase(it);
    stopped.erase(name);
    startEvents.erase(name);
    stopEvents.erase(name);
  }
}

void TimerGpu::checkError(cudaError_t result, const std::string& action) const {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Error after " << action << ": "
              << cudaGetErrorString(result) << std::endl;
  }
}