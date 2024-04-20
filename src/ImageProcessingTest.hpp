#pragma once
#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "Utils/CudaResourceManager.hpp"
#include "Utils/TimerBase.hpp"
#include "Utils/TimerCpu.hpp"
#include "Utils/TimerGpu.hpp"

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

  void compareMat(const cv::Mat& actual, const cv::Mat& expected) const;

  cv::Mat inputImage;
  std::vector<std::string> ignoreNames;
  const std::string actualProcessTimeName =
      "Actual Image Processing Time on CPU";
  std::shared_ptr<CudaResourceManager> resourceManager;

 private:
  const std::string smallImagePath = "assets\\scene_small.jpg";
  const std::string largeImagePath = "assets\\scene_large.jpg";
  std::string formatSizeMismatchMessage(const cv::Mat& actual,
                                        const cv::Mat& expected) const;
  std::string formatTypeMismatchMessage(const cv::Mat& actual,
                                        const cv::Mat& expected) const;
  std::string formatChannelsMismatchMessage(const cv::Mat& actual,
                                            const cv::Mat& expected) const;
  std::string formatDepthMismatchMessage(const cv::Mat& actual,
                                         const cv::Mat& expected) const;
  std::string formatContentMismatchMessage(const cv::Mat& actual,
                                           const cv::Mat& expected) const;
};
