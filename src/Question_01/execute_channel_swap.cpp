#include "../ImageProcessingTest.h"

cv::Mat MakeQ1desiredMat(cv::Mat image) {
  cv::Mat referenceImage = image.clone();
  cv::cvtColor(referenceImage, referenceImage, cv::COLOR_BGR2RGB);
  return referenceImage;
}

cv::Mat channelSwapCpu(cv::Mat image) {
  cv::Mat result = cv::Mat::zeros(image.size(), image.type());
  int width = image.cols;
  int height = image.rows;

#pragma omp parallel for
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      cv::Vec3b pixel = image.at<cv::Vec3b>(j, i);
      result.at<cv::Vec3b>(j, i) = cv::Vec3b(pixel[2], pixel[1], pixel[0]);
    }
  }
  return result;
}

TEST_F(ImageProcessingTest, Question_01_CPU) {
  cv::Mat image = readAssetsImage(true);
  cv::Mat desiredImage = MakeQ1desiredMat(image);

  timerCpu->start();
  cv::Mat resultCpu = channelSwapCpu(image);
  timerCpu->stop();

  std::cout << std::format("[{}] CPU time: {:.2f} ms\n", getCurrentTestName(),
                           timerCpu->elapsedMilliseconds());

  MatCompareResult compareResult = compareMat(resultCpu, desiredImage);
  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}