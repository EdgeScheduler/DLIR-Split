#ifndef DRIVERS_H
#define DRIVERS_H

#include<string>

/// @brief drivers to onnxruntime
class Drivers
{
public:
    /// @brief CPU driver
    static std::string CPUDriver;

    /// @brief CUDA-GPU driver
    static std::string GPUDriver;
};

#endif // !DRIVERS_H