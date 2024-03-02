#pragma once
#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "Utils/TimerCpu.h"
#include "Utils/TimerGpu.h"

enum class MatCompareResult {
  kMatch = 0,
  kContentMismatch = -1,
  kSizeMismatch = -2,
  kTypeMismatch = -3,
  kUnknown = -4
};

class ImageProcessingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase(){};
  static void TearDownTestCase(){};
  void SetUp() override;
  void TearDown() override;
  cv::Mat readAssetsImage(bool isLageImage = false) const;
  const std::string& getAssetImagePath(bool isLarge = false) const;
  std::string getCurrentTestName() const;
  std::string getOutputDir() const;
  MatCompareResult compareMat(const cv::Mat& actual,
                              const cv::Mat& desired) const;

  std::shared_ptr<TimerCpu> timerCpu;
  std::shared_ptr<TimerGpu> timerGpu;

 private:
  const std::string smallImagePath = "assets\\scene_small.jpg";
  const std::string largeImagePath = "assets\\scene_large.jpg";
};
