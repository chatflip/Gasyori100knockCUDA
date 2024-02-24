#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>

int main(int argc, const char* argv[]) {
	std::string srcRoot = "assets\\";
	std::string dstRoot = "output\\tutorial\\";
	std::filesystem::create_directories(dstRoot);

	cv::Mat img = cv::imread(srcRoot + "imori.jpg", cv::IMREAD_COLOR);

	int width = img.rows;
	int height = img.cols;

	cv::Mat out = img.clone();

	for (int i = 0; i < width / 2; i++) {
		for (int j = 0; j < height / 2; j++) {
			unsigned char tmp = out.at<cv::Vec3b>(j, i)[0];
			out.at<cv::Vec3b>(j, i)[0] = out.at<cv::Vec3b>(j, i)[2];
			out.at<cv::Vec3b>(j, i)[2] = tmp;
		}
	}

	cv::imwrite(dstRoot + "out.jpg", out);
	cv::imshow("sample", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}
