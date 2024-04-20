#include "ImageProcessingTest.hpp"

void ImageProcessingTest::SetUp() {
  inputImage = readAssetsImage(true);
  ignoreNames.push_back(actualProcessTimeName);
  resourceManager = std::make_shared<CudaResourceManager>();
}

void ImageProcessingTest::TearDown() { ignoreNames.clear(); }

std::string ImageProcessingTest::getQuestionLogDir(int numQuestion) {
  std::string outputDir = std::format("log\\Question{:02d}", numQuestion);
  std::filesystem::create_directories(outputDir);
  return outputDir;
};

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

std::string ImageProcessingTest::getGtestLogDir() const {
  std::string outputDir = std::format("log\\{}", getCurrentTestName());
  std::filesystem::create_directories(outputDir);
  return outputDir;
};

std::string ImageProcessingTest::formatSizeMismatchMessage(
    const cv::Mat& actual, const cv::Mat& expected) const {
  cv::Size AcutalSize = actual.size();
  cv::Size ExpectedSize = expected.size();
  return std::format(
      "Mat sizes are not equal\n"
      "Actual:\twidth:{} height:{}\n"
      "Expected:\twidth:{} height:{}",
      AcutalSize.width, AcutalSize.height, ExpectedSize.width,
      ExpectedSize.height);
}

std::string ImageProcessingTest::formatTypeMismatchMessage(
    const cv::Mat& actual, const cv::Mat& expected) const {
  return std::format(
      "Mat types are not equal\n"
      "Actual:\t{}\n"
      "Expected:\t{}",
      actual.type(), expected.type());
}

std::string ImageProcessingTest::formatChannelsMismatchMessage(
    const cv::Mat& actual, const cv::Mat& expected) const {
  return std::format(
      "Mat num channels are not equal\n"
      "Actual:\t{}\n"
      "Expected:\t{}",
      actual.channels(), expected.channels());
}

std::string ImageProcessingTest::formatDepthMismatchMessage(
    const cv::Mat& actual, const cv::Mat& expected) const {
  return std::format(
      "Mat depth are not equal\n"
      "Actual:\t{}\n"
      "Expected:\t{}",
      actual.depth(), expected.depth());
}

std::string ImageProcessingTest::formatContentMismatchMessage(
    const cv::Mat& actual, const cv::Mat& expected) const {
  cv::Mat diff;
  cv::absdiff(expected, actual, diff);
  int numDiffPixels = cv::countNonZero(diff);
  double minDiff, maxDiff;
  cv::minMaxLoc(diff, &minDiff, &maxDiff);
  double avgDiff = cv::mean(diff)[0];
  return std::format(
      "Mat pixels are not equal\n"
      "Number of pixels with different intensities:\t{}\n"
      "Minimum intensity difference:\t{}\n"
      "Maximum intensity difference:\t{}\n"
      "Average intensity difference:\t{:.2f}",
      numDiffPixels, minDiff, maxDiff, avgDiff);
}

void ImageProcessingTest::compareMat(const cv::Mat& actual,
                                     const cv::Mat& expected) const {
  EXPECT_TRUE(actual.size() == expected.size())
      << formatSizeMismatchMessage(actual, expected);

  EXPECT_TRUE(actual.type() == expected.type())
      << formatTypeMismatchMessage(actual, expected);
  EXPECT_TRUE(actual.channels() == expected.channels())
      << formatChannelsMismatchMessage(actual, expected);
  EXPECT_TRUE(actual.depth() == expected.depth())
      << formatDepthMismatchMessage(actual, expected);

  cv::Mat diff;
  std::vector<cv::Mat> diffChannels;
  cv::absdiff(actual, expected, diff);
  cv::split(diff, diffChannels);

  for (const auto& channel : diffChannels) {
    if (cv::countNonZero(channel) != 0) {
      FAIL() << formatContentMismatchMessage(actual, expected);
    }
  }
  SUCCEED();
};
