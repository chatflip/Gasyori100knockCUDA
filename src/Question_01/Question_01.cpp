#include "../ImageProcessingTest.h"
#include "../Utils/UtilsCuda.cuh"
#include "channel_swap_cpu.hpp"
#include "channel_swap_gpu.cuh"

namespace {
int numQuestions = 1;
std::string logDir = ImageProcessingTest::getQuestionLogDir(numQuestions);

cv::Mat MakeQ1desiredMat(cv::Mat image) {
  cv::Mat referenceImage = image.clone();
  cv::cvtColor(referenceImage, referenceImage, cv::COLOR_BGR2RGB);
  return referenceImage;
}
}  // namespace

TEST_F(ImageProcessingTest, Question_01_cpu) {
  cv::Mat desiredImage = MakeQ1desiredMat(inputImage);
  std::shared_ptr<TimerCpu> cpuTimer = std::make_shared<TimerCpu>();

  cpuTimer->start(actualProcessTimeName);
  cv::Mat resultCpu = bgr2rgbCpuInplace(inputImage, cpuTimer);
  cpuTimer->stop(actualProcessTimeName);
  cpuTimer->recordAll();
  float elapsedTime = cpuTimer->getRecord(actualProcessTimeName);
  cpuTimer->popRecord(actualProcessTimeName);

  std::string header = cpuTimer->createHeader(getCurrentTestName());
  std::string footer = cpuTimer->createFooter(elapsedTime);
  std::string logPath = std::format("{}\\{}.log", logDir, getCurrentTestName());
  cpuTimer->writeToFile(logPath, header, footer);
  cpuTimer->print(header, footer);

  MatCompareResult compareResult = compareMat(resultCpu, desiredImage);
  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}

TEST_F(ImageProcessingTest, Question_01_gpu) {
  cv::Mat desiredImage = MakeQ1desiredMat(inputImage);
  std::shared_ptr<TimerCpu> cpuTimer = std::make_shared<TimerCpu>();
  std::shared_ptr<TimerGpu> gpuTimer = std::make_shared<TimerGpu>();
  cudaStream_t stream = createCudaStream();

  cpuTimer->start(actualProcessTimeName);
  cv::Mat resultGpu = bgr2rgbGpuInplace(inputImage, stream, cpuTimer, gpuTimer);
  cpuTimer->stop(actualProcessTimeName);
  cpuTimer->recordAll();
  float elapsedTime = cpuTimer->getRecord(actualProcessTimeName);
  cpuTimer->popRecord(actualProcessTimeName);
  gpuTimer->recordAll();
  gpuTimer->mergeRecords(*cpuTimer);

  std::string header = gpuTimer->createHeader(getCurrentTestName());
  std::string footer = gpuTimer->createFooter(elapsedTime);
  std::string logPath = std::format("{}\\{}.log", logDir, getCurrentTestName());
  gpuTimer->writeToFile(logPath, header, footer);
  gpuTimer->print(header, footer);

  MatCompareResult compareResult = compareMat(resultGpu, desiredImage);
  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}

TEST_F(ImageProcessingTest, Question_01_gpu_thrust) {
  cv::Mat desiredImage = MakeQ1desiredMat(inputImage);
  std::shared_ptr<TimerCpu> cpuTimer = std::make_shared<TimerCpu>();
  std::shared_ptr<TimerGpu> gpuTimer = std::make_shared<TimerGpu>();
  cudaStream_t stream = createCudaStream();

  cpuTimer->start(actualProcessTimeName);
  cv::Mat resultGpu = bgr2rgbGpuThrust(inputImage, stream, cpuTimer, gpuTimer);
  cpuTimer->stop(actualProcessTimeName);
  cpuTimer->recordAll();
  float elapsedTime = cpuTimer->getRecord(actualProcessTimeName);
  cpuTimer->popRecord(actualProcessTimeName);
  gpuTimer->recordAll();
  gpuTimer->mergeRecords(*cpuTimer);

  std::string header = gpuTimer->createHeader(getCurrentTestName());
  std::string footer = gpuTimer->createFooter(elapsedTime);
  std::string logPath = std::format("{}\\{}.log", logDir, getCurrentTestName());
  gpuTimer->writeToFile(logPath, header, footer);
  gpuTimer->print(header, footer);

  MatCompareResult compareResult = compareMat(resultGpu, desiredImage);
  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}

TEST_F(ImageProcessingTest, Question_01_gpu_texture) {
  cv::Mat desiredImage = MakeQ1desiredMat(inputImage);
  std::shared_ptr<TimerCpu> cpuTimer = std::make_shared<TimerCpu>();
  std::shared_ptr<TimerGpu> gpuTimer = std::make_shared<TimerGpu>();
  cudaStream_t stream = createCudaStream();

  cpuTimer->start(actualProcessTimeName);
  cv::Mat resultGpu = bgr2rgbGpuTexture(inputImage, stream, cpuTimer, gpuTimer);
  cpuTimer->stop(actualProcessTimeName);
  cpuTimer->recordAll();
  float elapsedTime = cpuTimer->getRecord(actualProcessTimeName);
  cpuTimer->popRecord(actualProcessTimeName);
  gpuTimer->recordAll();
  gpuTimer->mergeRecords(*cpuTimer);

  std::string header = gpuTimer->createHeader(getCurrentTestName());
  std::string footer = gpuTimer->createFooter(elapsedTime);
  std::string logPath = std::format("{}\\{}.log", logDir, getCurrentTestName());
  gpuTimer->writeToFile(logPath, header, footer);
  gpuTimer->print(header, footer);

  MatCompareResult compareResult = compareMat(resultGpu, desiredImage);
  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}