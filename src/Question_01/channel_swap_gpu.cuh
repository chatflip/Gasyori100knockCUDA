#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>

#include <opencv2/opencv.hpp>

#include "../Utils/TimerCpu.hpp"
#include "../Utils/TimerGpu.hpp"

__global__ void bgr2rgbKernel(uchar* input, uchar* output, int width,
                              int height);

__global__ void bgr2rgbInplaceKernel(uchar* input, int width, int height);

__global__ void bgr2rgbTextureKernel(cudaTextureObject_t texObj, uchar4* output,
                                     int width, int height);

cv::Mat bgr2rgbGpuInplace(cv::Mat image, std::vector<cudaStream_t> streams,
                          std::shared_ptr<TimerCpu> cpuTimer,
                          std::shared_ptr<TimerGpu> gpuTimer);

cv::Mat bgr2rgbGpuThrust(cv::Mat image, std::vector<cudaStream_t> streams,
                         std::shared_ptr<TimerCpu> cpuTimer,
                         std::shared_ptr<TimerGpu> gpuTimer);

cv::Mat bgr2rgbGpuTexture(cv::Mat image, std::vector<cudaStream_t> streams,
                          std::shared_ptr<TimerCpu> cpuTimer,
                          std::shared_ptr<TimerGpu> gpuTimer);