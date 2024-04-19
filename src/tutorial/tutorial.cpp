#include "../ImageProcessingTest.hpp"

namespace {
cv::Mat drawRedLeftTopHalf(cv::Mat image, std::shared_ptr<TimerCpu> timer) {
  timer->start("Allocate Result Memory");
  cv::Mat result = image.clone();
  timer->stop("Allocate Result Memory");

  timer->start("Execute Image Processing");
  int width = image.cols;
  int height = image.rows;
  for (int j = 0; j < height / 2; j++) {
    for (int i = 0; i < width / 2; i++) {
      cv::Vec3b *src = result.ptr<cv::Vec3b>(j, i);
      uchar tmp = src[0][2];
      src[0][2] = src[0][0];
      src[0][0] = tmp;
    }
  }
  timer->stop("Execute Image Processing");
  return result;
}
}  // namespace

TEST_F(ImageProcessingTest, Tutorial) {
  std::shared_ptr<TimerCpu> cpuTimer = std::make_shared<TimerCpu>();

  cpuTimer->start(actualProcessTimeName);
  cv::Mat result = drawRedLeftTopHalf(inputImage, cpuTimer);
  cpuTimer->stop(actualProcessTimeName);
  cpuTimer->recordAll();
  float elapsedTime = cpuTimer->getRecord(actualProcessTimeName);
  cpuTimer->popRecord(actualProcessTimeName);

  std::string header = cpuTimer->createHeader(getCurrentTestName());
  std::string footer = cpuTimer->createFooter(elapsedTime);
  std::string logPath = std::format("{}\\benckmark.log", getGtestLogDir());
  cpuTimer->writeToFile(logPath, header, footer);
  cpuTimer->print(header, footer);

  SUCCEED();
}
