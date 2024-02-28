#include "ImageProcessingTest.h"

void ImageProcessingTest::SetUp() { timerCpu = std::make_shared<TimerCpu>(); }

void ImageProcessingTest::TearDown() { timerCpu->reset(); }

const std::string& ImageProcessingTest::getAssetImagePath(bool isLarge) const {
  return isLarge ? largeImagePath : smallImagePath;
};

cv::Mat ImageProcessingTest::readAssetsImage(bool isLageImage) const {
  cv::Mat image = cv::imread(getAssetImagePath(isLageImage), cv::IMREAD_COLOR);
  return image.clone();
};

std::string ImageProcessingTest::getCurrentTestName() const {
  const ::testing::TestInfo* const testInfo =
      ::testing::UnitTest::GetInstance()->current_test_info();
  return testInfo->name();
};

std::string ImageProcessingTest::getOutputDir() const {
  std::string outputDir = std::format("output\\{}", getCurrentTestName());
  std::filesystem::create_directories(outputDir);
  return outputDir;
};