#include <opencv2/opencv.hpp>

#include "../Utils/TimerCpu.hpp"

cv::Mat bgr2rgbCpuInplace(cv::Mat image, std::shared_ptr<TimerCpu> timer);