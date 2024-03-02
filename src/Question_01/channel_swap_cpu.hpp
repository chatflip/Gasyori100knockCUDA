#include <opencv2/opencv.hpp>

#include "../Utils/TimerBase.h"

cv::Mat bgr2rgbCpu(cv::Mat image, std::shared_ptr<TimerBase> timer);