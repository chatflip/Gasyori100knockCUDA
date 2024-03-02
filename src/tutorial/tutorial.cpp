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

  cv::Mat image = readAssetsImage();
  cv::Mat result = drawRedLeftTopHalf(image, timerCpu);

  float elapsedTime = timerCpu->calculateTotal(ignoreNames);
  std::string header = timerCpu->createHeader(getCurrentTestName());
  std::string footer = timerCpu->createFooter(elapsedTime);

  timerCpu->print(header, footer);
  timerCpu->writeToFile(std::format("{}\\benckmark.txt", getOutputDir()),
                        header, footer);
  SUCCEED();
}
