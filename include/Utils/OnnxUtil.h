#ifndef __ONNXUTIL_H__
#define __ONNXUTIL_H__

#include <iostream>
#include "../../library/onnx.proto3.pb.h"
#include <fstream>
#include <filesystem>

static class OnnxUtil
{
    public:
        /// @brief 
        /// @return 
        static onnx::ModelProto load(const std::filesystem::path &onnx_path="");
} onnxUtil;
#endif // __ONNXUTIL_H__