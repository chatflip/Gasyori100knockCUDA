#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include "../Utils/TimerBase.h"
#include "thrust/copy.h"
#include "thrust/device_vector.h"

__global__ void bgr2rgbKernel(unsigned char* input, unsigned char* output,
                              int width, int height);
cv::Mat bgr2rgbGpu(cv::Mat image, std::shared_ptr<TimerBase> timer);

cv::Mat bgr2rgbGpuThrust(cv::Mat image, std::shared_ptr<TimerBase> timer);