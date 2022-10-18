#ifndef __DRIVERS_H__
#define __DRIVERS_H__

#include <onnxruntime_cxx_api.h>

/// @brief drivers to onnxruntime
namespace Drivers
{
    /// @brief drivers to onnxruntime-CUDA
    class GPU_CUDA
    {
    public:
        /// @brief GPU-0 driver
        static OrtCUDAProviderOptions GPU0;
        /// @brief GPU-1 driver
        static OrtCUDAProviderOptions GPU1;
    };
};

#endif // __DRIVERS_H__