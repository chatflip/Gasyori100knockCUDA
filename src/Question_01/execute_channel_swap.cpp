#include "../ImageProcessingTest.h"

namespace {
cv::Mat MakeQ1desiredMat(cv::Mat image) {
  cv::Mat referenceImage = image.clone();
  cv::cvtColor(referenceImage, referenceImage, cv::COLOR_BGR2RGB);
  return referenceImage;
}
}  // namespace

cv::Mat bgr2rgbCpu(cv::Mat image) {
  // in-place channel swap
  int width = image.cols;
  int height = image.rows;

#pragma omp parallel for
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      cv::Vec3b *src = image.ptr<cv::Vec3b>(j, i);
      uchar tmp = src[0][2];
      src[0][2] = src[0][0];
      src[0][0] = tmp;
    }
  }
  return image;
}

TEST_F(ImageProcessingTest, Question_01_CPU) {
  cv::Mat image = readAssetsImage(true);
  cv::Mat desiredImage = MakeQ1desiredMat(image);

  timerCpu->start("aaa");
  cv::Mat resultCpu = bgr2rgbCpu(image);
  timerCpu->stop("aaa");

  std::cout << std::format("[{}] CPU time: {:.2f} ms\n", getCurrentTestName(),
                           timerCpu->elapsedMilliseconds("aaa"));

  std::string outPath = std::format("{}\\question_01_cpu.png", getOutputDir());
  cv::imwrite(outPath, resultCpu);
  timerCpu->print();

  MatCompareResult compareResult = compareMat(resultCpu, desiredImage);
  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}
