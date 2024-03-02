#include <cuda_runtime.h>

#include <iomanip>

#include "../ImageProcessingTest.h"

namespace {
cv::Mat MakeQ1desiredMat(cv::Mat image) {
  cv::Mat referenceImage = image.clone();
  cv::cvtColor(referenceImage, referenceImage, cv::COLOR_BGR2RGB);
  return referenceImage;
}
}  // namespace

__global__ void bgr2rgbKernel(unsigned char* input, unsigned char* output,
                              int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int idx = (y * width + x) * 3;  // 3 channels

  // BGR to RGB (simply swap B and R channels)
  uchar B = input[idx];
  uchar G = input[idx + 1];
  uchar R = input[idx + 2];

  output[idx] = R;
  output[idx + 1] = G;
  output[idx + 2] = B;
}

cv::Mat bgr2rgbGpu(cv::Mat image) {
  int width = image.cols;
  int height = image.rows;

  cv::Mat result = cv::Mat::zeros(height, width, image.type());

  uchar *d_input, *d_output;
  size_t numBytes = sizeof(uchar) * width * height * image.channels();

  // Allocate memory on gpu
  cudaMalloc(&d_input, numBytes);
  cudaMalloc(&d_output, numBytes);

  // copy host(cpu) to device(gpu)
  cudaMemcpy(d_input, image.data, numBytes, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  bgr2rgbKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
  // cudaDeviceSynchronize();
  cudaMemcpy(result.data, d_output, numBytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);

  return result;
}

TEST_F(ImageProcessingTest, Question_01_GPU) {
  cv::Mat image = readAssetsImage(true);
  cv::Mat desiredImage = MakeQ1desiredMat(image);

  timerCpu->start("aaa");
  cv::Mat resultGpu = bgr2rgbGpu(image);
  timerCpu->stop("aaa");

  std::cout << std::fixed << std::setprecision(2) << "[" << getCurrentTestName()
            << "] GPU time: " << timerCpu->elapsedMilliseconds("aaa") << "ms"
            << std::endl;

  std::string outPath = getOutputDir() + "\\question_01_gpu.png";
  cv::imwrite(outPath, resultGpu);

  MatCompareResult compareResult = compareMat(resultGpu, desiredImage);
  EXPECT_EQ(compareResult, MatCompareResult::kMatch);
}