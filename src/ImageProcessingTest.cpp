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

void ImageProcessingTest::analyzeMatDiff(
    const cv::Mat& actual, const cv::Mat& expected, double& maxAbsDiff,
    int& numDiffPixels, std::vector<cv::Point>& diffPositions) const {
  double minAbsVal, minRelVal;
  cv::Mat absDiff, relDiff;
  cv::absdiff(actual, expected, absDiff);
  cv::minMaxLoc(absDiff, &minAbsVal, &maxAbsDiff);
  numDiffPixels = cv::countNonZero(absDiff);
  cv::findNonZero(absDiff, diffPositions);
}

std::string ImageProcessingTest::formatContentMismatchMessage(
    const cv::Mat& actual, const cv::Mat& expected) const {
  double maxAbsDiff;
  int numDiffPixels;
  std::vector<cv::Point> diffPositions;
  analyzeMatDiff(actual, expected, maxAbsDiff, numDiffPixels, diffPositions);

  std::stringstream ss;
  for (const cv::Point diffPos : diffPositions) {
    auto x = diffPos.x;
    auto y = diffPos.y;
    uchar actualIntensity = actual.at<uchar>(y, x);
    uchar expectedIntensity = expected.at<uchar>(y, x);
    std::string meg = std::format("Actual({0},{1})={2}, Expected({0},{1})={3}",
                                  x, y, actualIntensity, expectedIntensity);
    ss << meg << std::endl;
  }

  return std::format(
      "Number of pixels with different intensities:\t{}\n"
      "Maximum intensity difference:\t{}\n"
      "Difference Details:\n{}",
      numDiffPixels, maxAbsDiff, ss.str());
}

void ImageProcessingTest::compareMatEqual(const cv::Mat& actual,
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

  int numNonZero = 0;
  for (const auto& channel : diffChannels) {
    numNonZero += cv::countNonZero(channel);
  }

  if (numNonZero == 0) {
    SUCCEED();
  } else {
    FAIL() << "Mat pixels are not equal\n"
           << formatContentMismatchMessage(actual, expected);
  }
};

void ImageProcessingTest::compareMatAlmostEqual(
    const cv::Mat& actual, const cv::Mat& expected, double thrMaxAbsDiff,
    double thrDiffPixelsPercent) const {
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

  int numNonZero = 0;
  for (const auto& channel : diffChannels) {
    numNonZero += cv::countNonZero(channel);
  }
  if (numNonZero == 0) {
    SUCCEED();
  }

  double maxAbsDiff;
  int numDiffPixels;
  std::vector<cv::Point> diffPositions;
  analyzeMatDiff(actual, expected, maxAbsDiff, numDiffPixels, diffPositions);

  double diffPixelsPercent = numDiffPixels * 100.0 / actual.size().area();

  if (maxAbsDiff <= thrMaxAbsDiff &&
      diffPixelsPercent <= thrDiffPixelsPercent) {
    std::cout << "Mat pixels are almost equal\n"
              << formatContentMismatchMessage(actual, expected) << std::endl;
    SUCCEED();
  } else {
    FAIL() << "Mat pixels are not equal\n"
           << formatContentMismatchMessage(actual, expected);
  }
}
