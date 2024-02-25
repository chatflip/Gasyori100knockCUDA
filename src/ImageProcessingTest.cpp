#include "ImageProcessingTest.h"

void ImageProcessingTest::SetUp() {

}

void ImageProcessingTest::TearDown() {

}


const std::string& ImageProcessingTest::getAssetImagePath() const {
	return assetImagePath;
};

cv::Mat ImageProcessingTest::readAssetsImage() const {
	cv::Mat image = cv::imread(getAssetImagePath(), cv::IMREAD_COLOR);
	return image.clone();
};
std::string ImageProcessingTest::getCurrentTestName() const {
	const ::testing::TestInfo* const testInfo =
		::testing::UnitTest::GetInstance()->current_test_info();
	return testInfo->name();
};

std::string ImageProcessingTest::getOutputDir() const {
	std::string outputDir = outputRoot + getCurrentTestName() + "\\";
	std::filesystem::create_directories(outputDir);
	return outputDir;
};