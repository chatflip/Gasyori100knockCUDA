#include "binarization.cuh"
/*
__global__ void bgr2grayMultiStreamKernel(uchar* input, uchar* output,
                                          int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int inIdx = (y * width + x) * 3;  // 3 channels
  int outIdx = y * width + x;
  float gray = __roundf(0.114f * input[inIdx] + 0.587f * input[inIdx + 1] +
                        0.299f * input[inIdx + 2]);
  output[outIdx] = static_cast<uchar>(gray);
}

cv::Mat bgr2grayGpuMultiStream(
    cv::Mat image, int numStreams,
    std::shared_ptr<CudaResourceManager> resourceManager,
    std::shared_ptr<TimerCpu> cpuTimer, std::shared_ptr<TimerGpu> gpuTimer) {
  std::vector<cudaStream_t> streams = resourceManager->getStreams(numStreams);

  cpuTimer->start("Allocate Result Memory");
  int width = image.cols;
  int height = image.rows;
  cv::Mat result(height, width, CV_8UC1);
  cpuTimer->stop("Allocate Result Memory");

  cpuTimer->start("Prepare Async Streams");
  // Since d_inputs is on the device, it cannot be freed by the destructor of
  // shared_ptr. Therefore, there is no advantage to use shared_ptr, so it is
  // managed by vector
  std::vector<uchar*> d_inputs, d_outputs;
  int rowsPerStream = height / numStreams;
  int inBytesPerRow = width * image.channels() * sizeof(uchar);
  int inBytesPerStream = rowsPerStream * inBytesPerRow;
  int outBytesPerRow = width * result.channels() * sizeof(uchar);
  int outBytesPerStream = rowsPerStream * outBytesPerRow;
  cpuTimer->stop("Prepare Async Streams");

  for (int i = 0; i < numStreams; i++) {
    gpuTimer->start("Async Allocate Device Memory", streams.at(i));
    uchar *d_input, *d_output;
    cudaMallocAsync(&d_input, inBytesPerStream, streams.at(i));
    cudaMallocAsync(&d_output, outBytesPerStream, streams.at(i));
    d_inputs.push_back(d_input);
    d_outputs.push_back(d_output);
    gpuTimer->stop("Async Allocate Device Memory", streams.at(i), false);
  }

  // Transfer host to device
  for (int i = 0; i < numStreams; i++) {
    gpuTimer->start("Async Transfer Host to Device", streams.at(i));
    int offset = i * rowsPerStream * inBytesPerRow;
    cudaMemcpyAsync(d_inputs.at(i), image.data + offset, inBytesPerStream,
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
    bgr2grayMultiStreamKernel<<<gridSize, blockSize, sharedMemSizeByte,
                                streams.at(i)>>>(
        d_inputs.at(i), d_outputs.at(i), width, heightPerStream);
    gpuTimer->stop("Async Execute Cuda Kernel", streams.at(i), false);
  }

  // Transfer device to host
  for (int i = 0; i < numStreams; i++) {
    gpuTimer->start("Async Transfer Device to Host Memory", streams.at(i));
    int offset = i * rowsPerStream * outBytesPerRow;
    cudaMemcpyAsync(result.data + offset, d_outputs.at(i), outBytesPerStream,
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
*/