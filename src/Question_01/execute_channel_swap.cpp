#include "../ImageProcessingTest.h"

namespace {
cv::Mat MakeQ1desiredMat(cv::Mat image) {
  cv::Mat referenceImage = image.clone();
  cv::cvtColor(referenceImage, referenceImage, cv::COLOR_BGR2RGB);
  return referenceImage;
}
}  // namespace

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

TEST_F(ImageProcessingTest, Question_01_CPU) {
  std::vector<std::string> ignoreNames = {"Allocate Destination Memory"};

  cv::Mat image = readAssetsImage(true);
  cv::Mat desiredImage = MakeQ1desiredMat(image);
  cv::Mat resultCpu = bgr2rgbCpu(image, timerCpu);

  float elapsedTime = timerCpu->calculateTotal(ignoreNames);
  std::string header = timerCpu->createHeader(getCurrentTestName());
  std::string footer = timerCpu->createFooter(elapsedTime);

  timerCpu->print(header, footer);
  timerCpu->writeToFile(std::format("{}\\benckmark_cpu.txt", getOutputDir()),
                        header, footer);

  MatCompareResult compareResult = compareMat(resultCpu, desiredImage);
  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}
