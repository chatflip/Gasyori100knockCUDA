
#include "channel_swap_gpu.cuh"

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

cv::Mat bgr2rgbGpu(cv::Mat image, std::shared_ptr<TimerBase> timer) {
  timer->start("Allocate Destination Memory");
  cv::Mat result = image.clone();
  timer->stop("Allocate Destination Memory");

  timer->start("Allocate Device Memory");
  int width = image.cols;
  int height = image.rows;
  uchar *d_input, *d_output;
  size_t numBytes = sizeof(uchar) * width * height * image.channels();

  // Allocate memory on gpu
  cudaMalloc(&d_input, numBytes);
  cudaMalloc(&d_output, numBytes);
  timer->stop("Allocate Device Memory");

  timer->start("Transfer Host to Device Memory");
  // copy host(cpu) to device(gpu)
  cudaMemcpy(d_input, image.data, numBytes, cudaMemcpyHostToDevice);
  timer->stop("Transfer Host to Device Memory");

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  timer->start("Execute Cuda Kernel");
  bgr2rgbKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
  timer->stop("Execute Cuda Kernel");

  timer->start("Transfer Device to Host Memory");
  cudaMemcpy(result.data, d_output, numBytes, cudaMemcpyDeviceToHost);
  timer->stop("Transfer Device to Host Memory");

  timer->start("Deallocate Device Memory");
  cudaFree(d_input);
  cudaFree(d_output);
  timer->stop("Deallocate Device Memory");

  return result;
}
