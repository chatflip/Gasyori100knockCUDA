#include <cuda_runtime.h>

/// <summary>
/// Create a CUDA stream
/// It is very slow(about 100ms) and has nothing to do with the image processing
/// itself, so I made it a separate function.
/// </summary>
/// <returns>cudaStream_t for Async</returns>
cudaStream_t createCudaStream();