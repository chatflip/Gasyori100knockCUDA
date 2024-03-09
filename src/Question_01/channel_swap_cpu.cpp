#include "channel_swap_cpu.hpp"

cv::Mat bgr2rgbCpuInplace(cv::Mat image, std::shared_ptr<TimerCpu> timer) {
  timer->start("Allocate Result Memory");
  cv::Mat result = image.clone();
  timer->stop("Allocate Result Memory");

  timer->start("Execute Image Processing");
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
  timer->stop("Execute Image Processing");
  return result;
}
