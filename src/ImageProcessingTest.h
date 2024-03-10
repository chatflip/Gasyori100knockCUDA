#pragma once
#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "Utils/TimerBase.h"
#include "Utils/TimerCpu.h"
#include "Utils/TimerGpu.cuh"

enum class MatCompareResult {
  kMatch = 0,
  kContentMismatch = -1,
  kSizeMismatch = -2,
  kTypeMismatch = -3,
  kChannelMismatch = -4,
  kUnknown = -5
};

class ImageProcessingTest : public ::testing::Test {
 public:
  static std::string getQuestionLogDir(int numQuestion);

 protected:
  static void SetUpTestCase(){};
  static void TearDownTestCase(){};
  void SetUp() override;
  void TearDown() override;
  cv::Mat readAssetsImage(bool isLageImage = false) const;
  const std::string& getAssetImagePath(bool isLarge = false) const;
  std::string getCurrentTestName() const;
  std::string getGtestLogDir() const;

  MatCompareResult compareMat(const cv::Mat& actual,
                              const cv::Mat& desired) const;
  std::string createHeader(const std::string& testName) const;

  cv::Mat inputImage;
  std::vector<std::string> ignoreNames;
  const std::string actualProcessTimeName =
      "Actual Image Processing Time on CPU";

 private:
  const std::string smallImagePath = "assets\\scene_small.jpg";
  const std::string largeImagePath = "assets\\scene_large.jpg";
};
