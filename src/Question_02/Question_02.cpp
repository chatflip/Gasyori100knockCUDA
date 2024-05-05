#include "../ImageProcessingTest.hpp"
#include "bgr2gray.cuh"

namespace {
int numQuestions = 2;
double thrMaxAbsDiff = 1;
double thrDiffPixelsPercent = 0.01;

std::string logDir = ImageProcessingTest::getQuestionLogDir(numQuestions);
cv::Mat MakeQ2desiredMat(cv::Mat image) {
  cv::Mat referenceImage = image.clone();
  cv::cvtColor(referenceImage, referenceImage, cv::COLOR_BGR2GRAY);
  return referenceImage;
}

cv::Mat bgr2grayCpu(cv::Mat image, std::shared_ptr<TimerCpu> timer) {
  timer->start("Allocate Result Memory");
  int width = image.cols;
  int height = image.rows;
  cv::Mat result(height, width, CV_8UC1);
  timer->stop("Allocate Result Memory");

  timer->start("Execute Image Processing");
#pragma omp parallel for collapse(2)
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      cv::Vec3b* src = image.ptr<cv::Vec3b>(j, i);
      uchar* dst = result.ptr<uchar>(j, i);
      // NOTE: This implementation does not strictly match OpenCV because
      // OpenCV's internal implementation performs calculations by bit shifting
      uchar gray =
          round(0.114f * src[0][0] + 0.587f * src[0][1] + 0.299f * src[0][2]);
      *dst = gray;
    }
  }
  timer->stop("Execute Image Processing");
  return result;
}

}  // namespace
TEST_F(ImageProcessingTest, Question_02_cpu) {
  cv::Mat desiredImage = MakeQ2desiredMat(inputImage);
  std::shared_ptr<TimerCpu> cpuTimer = std::make_shared<TimerCpu>();

  cpuTimer->start(actualProcessTimeName);
  cv::Mat resultCpu = bgr2grayCpu(inputImage, cpuTimer);
  cpuTimer->stop(actualProcessTimeName);
  cpuTimer->recordAll();
  float elapsedTime = cpuTimer->getRecord(actualProcessTimeName);
  cpuTimer->popRecord(actualProcessTimeName);

  std::string header = cpuTimer->createHeader(getCurrentTestName());
  std::string footer = cpuTimer->createFooter(elapsedTime);
  std::string logPath = std::format("{}\\{}.log", logDir, getCurrentTestName());
  cpuTimer->writeToFile(logPath, header, footer);
  cpuTimer->print(header, footer);

  compareMatAlmostEqual(resultCpu, desiredImage, thrMaxAbsDiff,
                        thrDiffPixelsPercent);
}

TEST_F(ImageProcessingTest, Question_02_gpu) {
  cv::Mat desiredImage = MakeQ2desiredMat(inputImage);
  std::shared_ptr<TimerCpu> cpuTimer = std::make_shared<TimerCpu>();
  std::shared_ptr<TimerGpu> gpuTimer = std::make_shared<TimerGpu>();
  int numStreams = 8;

  // warm up
  bgr2grayGpuMultiStream(dummyImage, numStreams, resourceManager,
                         std::make_shared<TimerCpu>(),
                         std::make_shared<TimerGpu>());

  cpuTimer->start(actualProcessTimeName);
  cv::Mat resultGpu = bgr2grayGpuMultiStream(
      inputImage, numStreams, resourceManager, cpuTimer, gpuTimer);
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

  compareMatAlmostEqual(resultGpu, desiredImage, thrMaxAbsDiff,
                        thrDiffPixelsPercent);
}
