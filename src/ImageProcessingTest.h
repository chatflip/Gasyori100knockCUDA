#pragma once
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>

class ImageProcessingTest : public ::testing::Test
{
protected:
	void SetUp() override;
	void TearDown() override;
	cv::Mat readAssetsImage() const;
	const std::string& getAssetImagePath() const;
	std::string getCurrentTestName() const;
	std::string getOutputDir() const;

private:
	const std::string assetImagePath = "assets\\imori.jpg";
	const std::string outputRoot = "output\\";
};

