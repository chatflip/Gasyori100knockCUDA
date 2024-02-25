#include "../ImageProcessingTest.h"

TEST_F(ImageProcessingTest, Tutorial) {
	cv::Mat image = readAssetsImage();

	timerCpu->start();

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
	timerCpu->stop();

	std::cout << std::format("[{}] CPU time: {:.2f} ms\n", getCurrentTestName(), timerCpu->elapsedMilliseconds());

	std::string outPath = std::format("{}\\out.png", getOutputDir());
	cv::imwrite(outPath, out);

	// cv::imshow("sample", out);
	// cv::waitKey(0);
	// cv::destroyAllWindows();

	SUCCEED();
}
