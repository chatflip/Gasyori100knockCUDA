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
  cv::Mat readDummyImage(bool isLageImage = false) const;
  const std::string& getAssetImagePath(bool isLarge = false) const;
  std::string getCurrentTestName() const;
  std::string getGtestLogDir() const;

  void compareMatEqual(const cv::Mat& actual, const cv::Mat& expected) const;
  void compareMatAlmostEqual(const cv::Mat& actual, const cv::Mat& expected,
                             double thrMaxAbsDiff,
                             double thrDiffPixelsPercent) const;

  cv::Mat inputImage, dummyImage;
  std::vector<std::string> ignoreNames;
  const std::string actualProcessTimeName =
      "Actual Image Processing Time on CPU";
  std::shared_ptr<CudaResourceManager> resourceManager;

 private:
  const std::string smallImagePath = "assets\\scene_small.jpg";
  const std::string largeImagePath = "assets\\scene_large.jpg";
  void analyzeMatDiff(const cv::Mat& actual, const cv::Mat& expected,
                      double& maxAbsDiff, int& numDiffPixels,
                      std::vector<cv::Point>& diffPositions) const;
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
