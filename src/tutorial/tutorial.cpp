#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>


#include "../ImageProcessingTest.h"


TEST_F(ImageProcessingTest, Tutorial) {
	std::string dstRoot = getOutputDir() + "tutorial\\";
	std::filesystem::create_directories(dstRoot);

	cv::Mat image = readAssetsImage();

	int width = image.rows;
	int height = image.cols;

	cv::Mat out = image.clone();

	for (int i = 0; i < width / 2; i++) {
		for (int j = 0; j < height / 2; j++) {
			unsigned char tmp = out.at<cv::Vec3b>(j, i)[0];
			out.at<cv::Vec3b>(j, i)[0] = out.at<cv::Vec3b>(j, i)[2];
			out.at<cv::Vec3b>(j, i)[2] = tmp;
		}
	}
	cv::imwrite(dstRoot + "out.jpg", out);

	// cv::imshow("sample", out);
	// cv::waitKey(0);
	// cv::destroyAllWindows();

	SUCCEED();
}
