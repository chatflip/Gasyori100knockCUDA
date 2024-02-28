#pragma once
#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "Utils/TimerCpu.h"

class ImageProcessingTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;
  cv::Mat readAssetsImage(bool isLageImage=false) const;
  const std::string& getAssetImagePath(bool isLarge=false) const;
  std::string getCurrentTestName() const;
  std::string getOutputDir() const;
  std::shared_ptr<TimerCpu> timerCpu;

 private:
  const std::string smallImagePath = "assets\\scene_small.jpg";
  const std::string largeImagePath = "assets\\scene_large.jpg";
};
