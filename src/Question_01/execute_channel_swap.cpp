#include "../ImageProcessingTest.h"

cv::Mat MakeQ1desiredMat(cv::Mat image) {
  cv::Mat referenceImage = image.clone();
  cv::cvtColor(referenceImage, referenceImage, cv::COLOR_BGR2RGB);
  return referenceImage;
}

TEST_F(ImageProcessingTest, Question_01_CPU) {
  cv::Mat image = readAssetsImage();
  cv::Mat desiredImage = MakeQ1desiredMat(image);

  cv::Mat result = image.clone();

  timerCpu->start();

  timerCpu->stop();
  std::cout << std::format("[{}] CPU time: {:.2f} ms\n", getCurrentTestName(),
                           timerCpu->elapsedMilliseconds());

  MatCompareResult compareResult = compareMat(result, desiredImage);

  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}