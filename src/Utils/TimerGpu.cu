#include "TimerGpu.h"

TimerGpu::TimerGpu() {}

TimerGpu::~TimerGpu() {
    reset();
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
}

void TimerGpu::start(const std::string& name) {
  started[name] = true;
  stopped[name] = false;
  cudaEvent_t startEvent;
  checkError(cudaEventCreate(&startEvent), "creating start event " + name);
  checkError(cudaEventRecord(startEvent, 0), "recording start event " + name);
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

double TimerGpu::elapsedMilliseconds(const std::string& name) const {
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
    std::cerr << "Timer Error:" << name
              << " Start and stop must be called before getting elapsed time."
              << std::endl;
    return -1.0;
  }

  float milliseconds = 0.0f;
  checkError(cudaEventElapsedTime(&milliseconds, startEvents.at(name),
                                  stopEvents.at(name)),
             "calculating elapsed time " + name);
  return milliseconds;
}

void TimerGpu::checkError(cudaError_t result, const std::string& action) const {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Error after " << action << ": "
              << cudaGetErrorString(result) << std::endl;
  }
}

void TimerGpu::writeToFile(const std::string& path) const {
  std::ofstream ofs(path.c_str());
  if (!ofs.is_open()) {
	std::cerr << "Error: could not open file " << path << std::endl;
	return;
  }

  for (auto& it : startEvents) {
      if (stopped.count(it.first) == 0) {
	  std::cerr << "Timer Error: " << it.first << " has not been stopped."
				<< std::endl;
	  continue;
	}

	ofs << it.first << ": " << elapsedMilliseconds(it.first) << " ms"
		 << std::endl;
  }
  ofs.close();
}