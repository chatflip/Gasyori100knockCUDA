#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include "../Utils/CudaResourceManager.hpp"
#include "../Utils/TimerCpu.hpp"
#include "../Utils/TimerGpu.hpp"

__global__ void binarizationMultiStreamKernel(uchar* input, uchar* output,
                                          int width, int height);

cv::Mat binarizationGpuMultiStream(
    cv::Mat image, int numStreams,
    std::shared_ptr<CudaResourceManager> resourceManager,
    std::shared_ptr<TimerCpu> cpuTimer, std::shared_ptr<TimerGpu> gpuTimer);
