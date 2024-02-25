#pragma once
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>

class ImageProcessingTest : public ::testing::Test
{
protected:
	void SetUp() override;
	void TearDown() override;

	const std::string& getAssetImagePath() const {
		return assetImagePath;
	};
	const std::string& getOutputDir() const {
		return outputDir;
	};
	cv::Mat readAssetsImage() const {
		cv::Mat image = cv::imread(getAssetImagePath(), cv::IMREAD_COLOR);
		return image.clone();
	};

private:
	const std::string assetImagePath = "assets\\imori.jpg";
	const std::string outputDir = "output\\";
};

