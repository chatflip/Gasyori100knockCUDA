#include "ImageProcessingTest.h"

void ImageProcessingTest::SetUp() {}

void ImageProcessingTest::TearDown() {}

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

std::string ImageProcessingTest::getLogDir() const {
  std::string outputDir = std::format("log\\{}", getCurrentTestName());
  std::filesystem::create_directories(outputDir);
  return outputDir;
};

std::string ImageProcessingTest::getLogDir(int numQuestion) const {
  std::string outputDir = std::format("log\\Question{:02d}", numQuestion);
  std::filesystem::create_directories(outputDir);
  return outputDir;
};

MatCompareResult ImageProcessingTest::compareMat(const cv::Mat& actual,
                                                 const cv::Mat& desired) const {
  using enum MatCompareResult;
  if (actual.size() != desired.size()) {
    return kSizeMismatch;
  }
  if (actual.type() != desired.type()) {
    return kTypeMismatch;
  }
  if (actual.channels() != desired.channels()) {
    return kChannelMismatch;
  }
  cv::Mat diff;
  std::vector<cv::Mat> diffChannels;
  cv::absdiff(actual, desired, diff);
  cv::split(diff, diffChannels);

  for (const auto& channel : diffChannels) {
    if (cv::countNonZero(channel) != 0) {
      return kContentMismatch;
    }
  }
  return kMatch;
};
