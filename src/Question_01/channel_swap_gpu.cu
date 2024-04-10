
#include "channel_swap_gpu.cuh"

__global__ void bgr2rgbKernel(uchar* input, uchar* output, int width,
                              int height) {
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

__global__ void bgr2rgbInplaceKernel(uchar* input, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int idx = (y * width + x) * 3;  // 3 channels

  // BGR to RGB (simply swap B and R channels)
  uchar temp = input[idx];
  input[idx] = input[idx + 2];
  input[idx + 2] = temp;
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

cv::Mat bgr2rgbGpuMultiStream(
    cv::Mat image, std::shared_ptr<CudaResourceManager> resourceManager,
    std::shared_ptr<TimerCpu> cpuTimer, std::shared_ptr<TimerGpu> gpuTimer) {
  std::vector<cudaStream_t> streams = resourceManager->getStreams();

  cpuTimer->start("Allocate Result Memory");
  cv::Mat result = image.clone();
  cpuTimer->stop("Allocate Result Memory");

  gpuTimer->start("Async Allocate Device Memory", streams.at(0));
  int width = image.cols;
  int height = image.rows;
  uchar* d_input;
  size_t numTensor = width * height * image.channels();
  size_t numBytes = sizeof(uchar) * numTensor;
  // Allocate memory on gpu
  cudaMallocAsync(&d_input, numBytes, streams.at(0));
  gpuTimer->stop("Async Allocate Device Memory", streams.at(0), false);

  gpuTimer->start("Async Transfer Host to Device", streams.at(0));
  // copy host(cpu) to device(gpu)
  cudaMemcpyAsync(d_input, image.data, numBytes, cudaMemcpyHostToDevice,
                  streams.at(0));
  gpuTimer->stop("Async Transfer Host to Device", streams.at(0), false);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  gpuTimer->start("Async Execute Cuda Kernel", streams.at(0));
  size_t sharedMemSizeByte = 0;
  bgr2rgbInplaceKernel<<<gridSize, blockSize, sharedMemSizeByte,
                         streams.at(0)>>>(d_input, width, height);
  gpuTimer->stop("Async Execute Cuda Kernel", streams.at(0), false);

  gpuTimer->start("Async Transfer Device to Host Memory", streams.at(0));
  cudaMemcpyAsync(result.data, d_input, numBytes, cudaMemcpyDeviceToHost,
                  streams.at(0));
  gpuTimer->stop("Async Transfer Device to Host Memory", streams.at(0), false);

  gpuTimer->start("Async Deallocate Device Memory", streams.at(0));
  cudaFreeAsync(d_input, streams.at(0));
  gpuTimer->stop("Async Deallocate Device Memory", streams.at(0), false);

  gpuTimer->start("Wait For GPU Execution");
  cudaStreamSynchronize(streams.at(0));
  cudaStreamDestroy(streams.at(0));
  gpuTimer->stop("Wait For GPU Execution");

  return result;
}

cv::Mat bgr2rgbGpuThrust(cv::Mat image,
                         std::shared_ptr<CudaResourceManager> resourceManager,
                         std::shared_ptr<TimerCpu> cpuTimer,
                         std::shared_ptr<TimerGpu> gpuTimer) {
  std::vector<cudaStream_t> streams = resourceManager->getStreams();

  cpuTimer->start("Allocate Result Memory");
  int width = image.cols;
  int height = image.rows;
  cv::Mat result(height, width, image.type());
  cpuTimer->stop("Allocate Result Memory");

  gpuTimer->start("Allocate And Transfer Memory");
  size_t numTensor = width * height * image.channels();

  // Allocate memory on gpu
  thrust::device_vector<uchar> d_input(image.data, image.data + numTensor);
  thrust::device_vector<uchar> d_output(numTensor);
  gpuTimer->stop("Allocate And Transfer Memory", streams.at(0), false);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  gpuTimer->start("Async Execute Cuda Kernel");
  size_t sharedMemSizeByte = 0;
  bgr2rgbKernel<<<gridSize, blockSize, sharedMemSizeByte, streams.at(0)>>>(
      thrust::raw_pointer_cast(d_input.data()),
      thrust::raw_pointer_cast(d_output.data()), width, height);
  gpuTimer->stop("Async Execute Cuda Kernel", streams.at(0), false);

  gpuTimer->start("Async Transfer Device to Host Memory");
  cudaMemcpyAsync(result.data, thrust::raw_pointer_cast(d_output.data()),
                  numTensor * sizeof(uchar), cudaMemcpyDeviceToHost,
                  streams.at(0));
  gpuTimer->stop("Async Transfer Device to Host Memory", streams.at(0), false);

  gpuTimer->start("Wait For GPU Execution");
  cudaStreamSynchronize(streams.at(0));
  cudaStreamDestroy(streams.at(0));
  gpuTimer->stop("Wait For GPU Execution");

  return result;
}

cv::Mat bgr2rgbGpuTexture(cv::Mat image,
                          std::shared_ptr<CudaResourceManager> resourceManager,
                          std::shared_ptr<TimerCpu> cpuTimer,
                          std::shared_ptr<TimerGpu> gpuTimer) {
  std::vector<cudaStream_t> streams = resourceManager->getStreams();

  cpuTimer->start("Allocate Result Memory");
  int width = image.cols;
  int height = image.rows;
  cv::Mat result(height, width, CV_8UC4);
  cpuTimer->stop("Allocate Result Memory");

  gpuTimer->start("Optimize Input Image For Texture Memory");
  cv::Mat inputBGRA;
  cv::cvtColor(image, inputBGRA, cv::COLOR_BGR2BGRA);
  gpuTimer->stop("Optimize Input Image For Texture Memory");

  gpuTimer->start("Allocate Device Memory");
  cudaArray* cuArray;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
  cudaMallocArray(&cuArray, &channelDesc, width, height);
  uchar4* d_output;
  size_t numBytes = sizeof(uchar4) * width * height;
  cudaMallocAsync(&d_output, numBytes, streams.at(0));
  gpuTimer->stop("Allocate Device Memory", streams.at(0), false);

  gpuTimer->start("Transfer Host to Device");
  cudaMemcpyToArray(cuArray, 0, 0, inputBGRA.data, inputBGRA.step * height,
                    cudaMemcpyHostToDevice);
  gpuTimer->stop("Transfer Host to Device");

  gpuTimer->start("Create Texture Memory");
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
  gpuTimer->stop("Create Texture Memory");

  gpuTimer->start("Execute Cuda Kernel");
  dim3 blockSize(16, 16);
  size_t sharedMemSizeByte = 0;
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
  bgr2rgbTextureKernel<<<gridSize, blockSize, sharedMemSizeByte,
                         streams.at(0)>>>(texObj, d_output, width, height);
  gpuTimer->stop("Execute Cuda Kernel");

  gpuTimer->start("Transfer Device to Host Memory");
  cudaMemcpyAsync(result.data, d_output, numBytes, cudaMemcpyDeviceToHost,
                  streams.at(0));
  gpuTimer->stop("Transfer Device to Host Memory", streams.at(0), false);

  gpuTimer->start("Deallocate Device Memory");
  cudaDestroyTextureObject(texObj);
  cudaFreeArray(cuArray);
  cudaFreeAsync(d_output, streams.at(0));
  gpuTimer->stop("Deallocate Device Memory");

  gpuTimer->start("Wait For GPU Execution");
  cudaStreamSynchronize(streams.at(0));
  cudaDeviceSynchronize();
  cudaStreamDestroy(streams.at(0));
  gpuTimer->stop("Wait For GPU Execution");

  gpuTimer->start("Restore Output Image");
  cv::cvtColor(result, result, cv::COLOR_BGRA2BGR);
  gpuTimer->stop("Restore Output Image");

  return result;
}