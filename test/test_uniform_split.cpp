#include <filesystem>
#include <iostream>
#include "ModelAnalyze/ModelAnalyzer.h"
#include "onnx/shape_inference/implementation.h"
#include "Utils/helper.h"
#include "Benchmark/evaluate_models.h"
#include "Utils/Optimizer.h"

using namespace std;
#define ONNX_NAMESPACE onnx

int main()
{
    ModelAnalyzer analyzer = ModelAnalyzer("vgg19");
    optimize(analyzer);

    return 0;
}