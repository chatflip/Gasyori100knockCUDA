
#include "bgr2rgb.cuh"

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

__global__ void bgr2rgbMultiStreamKernel(uchar* input, uchar* output, int width,
                                         int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  // BGR to RGB (simply swap B and R channels)
  int idx = (y * width + x) * 3;  // 3 channels
  output[idx] = input[idx + 2];
  output[idx + 1] = input[idx + 1];
  output[idx + 2] = input[idx + 0];
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
    cv::Mat image, int numStreams,
    std::shared_ptr<CudaResourceManager> resourceManager,
    std::shared_ptr<TimerCpu> cpuTimer, std::shared_ptr<TimerGpu> gpuTimer) {
  std::vector<cudaStream_t> streams = resourceManager->getStreams(numStreams);

  cpuTimer->start("Allocate Result Memory");
  int width = image.cols;
  int height = image.rows;
  cv::Mat result(height, width, image.type());
  cpuTimer->stop("Allocate Result Memory");

  cpuTimer->start("Prepare Async Streams");
  // Since d_inputs is on the device, it cannot be freed by the destructor of
  // shared_ptr. Therefore, there is no advantage to use shared_ptr, so it is
  // managed by vector
  std::vector<uchar*> d_inputs, d_outputs;
  int rowsPerStream = height / numStreams;
  int bytesPerRow = width * image.channels() * sizeof(uchar);
  int bytesPerStream = rowsPerStream * bytesPerRow;
  cpuTimer->stop("Prepare Async Streams");

  for (int i = 0; i < numStreams; i++) {
    gpuTimer->start("Async Allocate Device Memory", streams.at(i));
    uchar *d_input, *d_output;
    cudaMallocAsync(&d_input, bytesPerStream, streams.at(i));
    cudaMallocAsync(&d_output, bytesPerStream, streams.at(i));
    d_inputs.push_back(d_input);
    d_outputs.push_back(d_output);
    gpuTimer->stop("Async Allocate Device Memory", streams.at(i), false);
  }

  // Transfer host to device
  for (int i = 0; i < numStreams; i++) {
    gpuTimer->start("Async Transfer Host to Device", streams.at(i));
    int offset = i * rowsPerStream * bytesPerRow;
    cudaMemcpyAsync(d_inputs.at(i), image.data + offset, bytesPerStream,
                    cudaMemcpyHostToDevice, streams.at(i));
    gpuTimer->stop("Async Transfer Host to Device", streams.at(i), false);
  }

  // Execute kernel
  dim3 blockSize(16, 16);
  size_t sharedMemSizeByte = 0;
  for (int i = 0; i < numStreams; i++) {
    gpuTimer->start("Async Execute Cuda Kernel", streams.at(i));
    // Calculate the height of the current stream
    // The last stream may have more rows than the other streams
    int heightPerStream =
        (i == numStreams - 1) ? height - i * rowsPerStream : rowsPerStream;
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (heightPerStream + blockSize.y - 1) / blockSize.y);
    bgr2rgbMultiStreamKernel<<<gridSize, blockSize, sharedMemSizeByte,
                               streams.at(i)>>>(d_inputs.at(i), d_outputs.at(i),
                                                width, heightPerStream);
    gpuTimer->stop("Async Execute Cuda Kernel", streams.at(i), false);
  }

  // Transfer device to host
  for (int i = 0; i < numStreams; i++) {
    gpuTimer->start("Async Transfer Device to Host Memory", streams.at(i));
    int offset = i * rowsPerStream * bytesPerRow;
    cudaMemcpyAsync(result.data + offset, d_outputs.at(i), bytesPerStream,
                    cudaMemcpyDeviceToHost, streams.at(i));
    gpuTimer->stop("Async Transfer Device to Host Memory", streams.at(i),
                   false);
  }

  // Deallocate device memory
  for (int i = 0; i < numStreams; i++) {
    gpuTimer->start("Async Deallocate Device Memory", streams.at(i));
    cudaFreeAsync(d_inputs.at(i), streams.at(i));
    cudaFreeAsync(d_outputs.at(i), streams.at(i));
    gpuTimer->stop("Async Deallocate Device Memory", streams.at(i), false);
  }

  gpuTimer->start("Wait For GPU Execution");
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams.at(i));
  }
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
  gpuTimer->stop("Wait For GPU Execution");

  gpuTimer->start("Restore Output Image");
  cv::cvtColor(result, result, cv::COLOR_BGRA2BGR);
  gpuTimer->stop("Restore Output Image");

  return result;
}