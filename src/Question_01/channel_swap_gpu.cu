
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
  size_t numTensor = width * height * image.channels();
  size_t numBytes = sizeof(uchar) * numTensor;

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

cv::Mat bgr2rgbGpuThrust(cv::Mat image, std::shared_ptr<TimerBase> timer) {
  timer->start("Allocate Destination Memory");
  cv::Mat result = image.clone();
  timer->stop("Allocate Destination Memory");

  timer->start("Allocate Device Memory");
  int width = image.cols;
  int height = image.rows;
  size_t numTensor = width * height * image.channels();
  size_t numBytes = sizeof(uchar) * numTensor;

  // Allocate memory on gpu
  thrust::device_vector<uchar> d_input(numTensor);
  thrust::device_vector<uchar> d_output(numTensor);
  timer->stop("Allocate Device Memory");

  timer->start("Transfer Host to Device Memory");
  // copy host(cpu) to device(gpu)
  cudaMemcpy(thrust::raw_pointer_cast(d_input.data()), image.data, numTensor,
             cudaMemcpyHostToDevice);
  timer->stop("Transfer Host to Device Memory");

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  timer->start("Execute Cuda Kernel");
  bgr2rgbKernel<<<gridSize, blockSize>>>(
      thrust::raw_pointer_cast(d_input.data()),
      thrust::raw_pointer_cast(d_output.data()), width, height);
  timer->stop("Execute Cuda Kernel");

  timer->start("Transfer Device to Host Memory");
  cudaMemcpy(result.data, thrust::raw_pointer_cast(d_output.data()), numBytes,
             cudaMemcpyDeviceToHost);
  timer->stop("Transfer Device to Host Memory");
  return result;
}

__global__ void bgr2rgbTextureKernel(cudaTextureObject_t texObj, uchar4* output,
                                     int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  // Fetch pixel values from texture object (as BGR format)
  uchar4 bgraValue = tex2D<uchar4>(texObj, x, y);

  uchar4 rgbaValue;
  rgbaValue.x = bgraValue.z;  // R
  rgbaValue.y = bgraValue.y;  // G
  rgbaValue.z = bgraValue.x;  // B
  rgbaValue.w = bgraValue.w;  // A（Alpha channel remains unchanged.）

  output[y * width + x] = rgbaValue;
}

cv::Mat bgr2rgbGpuTexture(cv::Mat image, std::shared_ptr<TimerBase> timer) {
  timer->start("Allocate Destination Memory");
  int width = image.cols;
  int height = image.rows;
  size_t numBytes = sizeof(uchar4) * width * height;
  cv::Mat result(height, width, CV_8UC4);
  timer->stop("Allocate Destination Memory");

  timer->start("Optimize Input Image For Texture Memory");
  cv::Mat inputBGRA;
  cv::cvtColor(image, inputBGRA, cv::COLOR_BGR2BGRA);
  timer->stop("Optimize Input Image For Texture Memory");

  timer->start("Allocate Device Memory");
  cudaArray* cuArray;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
  cudaMallocArray(&cuArray, &channelDesc, width, height);
  uchar4* d_output;
  cudaMalloc(&d_output, numBytes);
  timer->stop("Allocate Device Memory");

  timer->start("Transfer Host to Device Memory");
  cudaMemcpyToArray(cuArray, 0, 0, inputBGRA.data, inputBGRA.step * height,
                    cudaMemcpyHostToDevice);
  timer->stop("Transfer Host to Device Memory");

  timer->start("Create Texture Memory");
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  timer->stop("Create Texture Memory");

  timer->start("Execute Cuda Kernel");
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
  bgr2rgbTextureKernel<<<gridSize, blockSize>>>(texObj, d_output, width,
                                                height);
  timer->stop("Execute Cuda Kernel");

  timer->start("Transfer Device to Host Memory");
  cudaMemcpy(result.data, d_output, numBytes, cudaMemcpyDeviceToHost);
  timer->stop("Transfer Device to Host Memory");

  timer->start("Wait For GPU Execution");
  cudaDeviceSynchronize();
  timer->stop("Wait For GPU Execution");

  timer->start("Deallocate Device Memory");
  cudaDestroyTextureObject(texObj);
  cudaFreeArray(cuArray);
  cudaFree(d_output);
  timer->stop("Deallocate Device Memory");

  timer->start("Restore Output Image");
  cv::cvtColor(result, result, cv::COLOR_BGRA2BGR);
  timer->stop("Restore Output Image");

  return result;
}