#include "channel_swap_cpu.hpp"

cv::Mat bgr2rgbCpu(cv::Mat image, std::shared_ptr<TimerBase> timer) {
  timer->start("Allocate Destination Memory");
  cv::Mat result = image.clone();
  timer->stop("Allocate Destination Memory");

  timer->start("Execute Process");
  // in-place channel swap
  int width = image.cols;
  int height = image.rows;

#pragma omp parallel for
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      cv::Vec3b *src = result.ptr<cv::Vec3b>(j, i);
      uchar tmp = src[0][2];
      src[0][2] = src[0][0];
      src[0][0] = tmp;
    }
  }
  timer->stop("Execute Process");
  return result;
}
