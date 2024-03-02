#include "../ImageProcessingTest.h"

cv::Mat drawRedLeftTopHalf(cv::Mat image, std::shared_ptr<TimerBase> timer) {
  timer->start("Allocate Destination Memory");
  cv::Mat result = image.clone();
  timer->stop("Allocate Destination Memory");

  timer->start("Execute Process");
  int width = image.cols;
  int height = image.rows;
  for (int j = 0; j < height / 2; j++) {
    for (int i = 0; i < width / 2; i++) {
      unsigned char tmp = result.at<cv::Vec3b>(j, i)[0];
      result.at<cv::Vec3b>(j, i)[0] = result.at<cv::Vec3b>(j, i)[2];
      result.at<cv::Vec3b>(j, i)[2] = tmp;
    }
  }
  timer->stop("Execute Process");
  return result;
}

TEST_F(ImageProcessingTest, Tutorial) {
  std::vector<std::string> ignoreNames = {"Allocate Destination Memory"};
  std::shared_ptr<TimerBase> timer = std::make_shared<TimerCpu>();

  cv::Mat image = readAssetsImage();
  cv::Mat result = drawRedLeftTopHalf(image, timer);

  float elapsedTime = timer->calculateTotal(ignoreNames);
  std::string header = timer->createHeader(getCurrentTestName());
  std::string footer = timer->createFooter(elapsedTime);

  timer->print(header, footer);
  std::string logPath = std::format("{}\\benckmark.txt", getOutputDir());
  timer->writeToFile(logPath, header, footer);

  SUCCEED();
}
