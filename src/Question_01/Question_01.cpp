#include "../ImageProcessingTest.h"
#include "channel_swap_cpu.hpp"
#include "channel_swap_gpu.cuh"

namespace {
cv::Mat MakeQ1desiredMat(cv::Mat image) {
  cv::Mat referenceImage = image.clone();
  cv::cvtColor(referenceImage, referenceImage, cv::COLOR_BGR2RGB);
  return referenceImage;
}
}  // namespace

TEST_F(ImageProcessingTest, Question_01_cpu) {
  int numQuestions = 1;
  std::vector<std::string> ignoreNames = {"Allocate Destination Memory"};
  std::shared_ptr<TimerBase> timer = std::make_shared<TimerCpu>();

  cv::Mat image = readAssetsImage(true);
  cv::Mat desiredImage = MakeQ1desiredMat(image);
  cv::Mat resultCpu = bgr2rgbCpu(image, timer);

  float elapsedTime = timer->calculateTotal(ignoreNames);
  std::string header = timer->createHeader(getCurrentTestName());
  std::string footer = timer->createFooter(elapsedTime);

  timer->print(header, footer);
  std::string logPath =
      std::format("{}\\benckmark_cpu.log", getLogDir(numQuestions));
  timer->writeToFile(logPath, header, footer);
  MatCompareResult compareResult = compareMat(resultCpu, desiredImage);

  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}

TEST_F(ImageProcessingTest, Question_01_gpu_raw) {
  int numQuestions = 1;
  std::vector<std::string> ignoreNames = {"Allocate Destination Memory"};
  std::shared_ptr<TimerBase> timer = std::make_shared<TimerGpu>();

  cv::Mat image = readAssetsImage(true);
  cv::Mat desiredImage = MakeQ1desiredMat(image);
  cv::Mat resultGpu = bgr2rgbGpu(image, timer);

  float elapsedTime = timer->calculateTotal(ignoreNames);
  std::string header = timer->createHeader(getCurrentTestName());
  std::string footer = timer->createFooter(elapsedTime);

  timer->print(header, footer);
  std::string logPath =
      std::format("{}\\benckmark_gpu_raw.log", getLogDir(numQuestions));
  timer->writeToFile(logPath, header, footer);
  MatCompareResult compareResult = compareMat(resultGpu, desiredImage);

  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}

TEST_F(ImageProcessingTest, Question_01_gpu_thrust) {
  int numQuestions = 1;
  std::vector<std::string> ignoreNames = {"Allocate Destination Memory"};
  std::shared_ptr<TimerBase> timer = std::make_shared<TimerGpu>();

  cv::Mat image = readAssetsImage(true);
  cv::Mat desiredImage = MakeQ1desiredMat(image);
  cv::Mat resultGpu = bgr2rgbGpuThrust(image, timer);

  float elapsedTime = timer->calculateTotal(ignoreNames);
  std::string header = timer->createHeader(getCurrentTestName());
  std::string footer = timer->createFooter(elapsedTime);

  timer->print(header, footer);
  std::string logPath =
      std::format("{}\\benckmark_gpu_thrust.log", getLogDir(numQuestions));
  timer->writeToFile(logPath, header, footer);
  MatCompareResult compareResult = compareMat(resultGpu, desiredImage);

  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}

TEST_F(ImageProcessingTest, Question_01_gpu_texture) {
  int numQuestions = 1;
  std::vector<std::string> ignoreNames = {
      "Allocate Destination Memory", "Optimize Input Image For Texture Memory",
      "Restore Output Image"};
  std::shared_ptr<TimerBase> timer = std::make_shared<TimerGpu>();

  cv::Mat image = readAssetsImage(true);
  cv::Mat desiredImage = MakeQ1desiredMat(image);
  cv::Mat resultGpu = bgr2rgbGpuTexture(image, timer);

  float elapsedTime = timer->calculateTotal(ignoreNames);
  std::string header = timer->createHeader(getCurrentTestName());
  std::string footer = timer->createFooter(elapsedTime);

  timer->print(header, footer);
  std::string logPath =
      std::format("{}\\benckmark_gpu_texture.log", getLogDir(numQuestions));
  timer->writeToFile(logPath, header, footer);
  MatCompareResult compareResult = compareMat(resultGpu, desiredImage);

  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}