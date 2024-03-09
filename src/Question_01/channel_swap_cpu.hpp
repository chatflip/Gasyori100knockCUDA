#include <opencv2/opencv.hpp>

#include "../Utils/TimerCpu.h"

cv::Mat bgr2rgbCpuInplace(cv::Mat image, std::shared_ptr<TimerCpu> timer);