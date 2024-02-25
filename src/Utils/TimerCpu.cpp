#include "TimerCpu.h"

TimerCpu::TimerCpu() {
	if (!QueryPerformanceFrequency(&frequency)) {
		std::cerr << "High precision counters are not supported by this system" << std::endl;
	}
}

void TimerCpu::reset() {
	startTime.QuadPart = 0;
	endTime.QuadPart = 0;
}

void TimerCpu::start() {
	QueryPerformanceCounter(&startTime);
}

void TimerCpu::stop() {
	QueryPerformanceCounter(&endTime);
}

double TimerCpu::elapsedMilliseconds() const {
	double interval = static_cast<double>(endTime.QuadPart - startTime.QuadPart) / static_cast<double>(frequency.QuadPart);
	return 1e3 * interval;
}
