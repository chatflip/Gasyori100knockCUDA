#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>

#include <opencv2/opencv.hpp>

#include "../Utils/CudaResourceManager.hpp"
#include "../Utils/TimerCpu.hpp"
#include "../Utils/TimerGpu.hpp"

__global__ void bgr2grayMultiStreamKernel(uchar* input, uchar* output,
                                          int width, int height);

cv::Mat bgr2grayGpuMultiStream(
    cv::Mat image, int numStreams,
    std::shared_ptr<CudaResourceManager> resourceManager,
    std::shared_ptr<TimerCpu> cpuTimer, std::shared_ptr<TimerGpu> gpuTimer);
