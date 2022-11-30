#include <filesystem>
#include <iostream>
#include<algorithm>
#include "ModelAnalyze/ModelAnalyzer.h"
#include "onnx/shape_inference/implementation.h"
#include "Utils/helper.h"
#include "Benchmark/evaluate_models.h"
#include "Utils/UniformOptimizer.h"
using namespace std;

int main()
{
    for(auto model_name: {"googlenet", "resnet50", "vgg19", "squeezenetv1"})
    {
        ModelAnalyzer analyzer = ModelAnalyzer(model_name);

        cout<<model_name<<":"<<endl;
        
        evam::TimeEvaluateChildModels_impl(model_name,-1,"RTX-2080Ti", false, 10);
        
        
        cout<<endl;
    }
    


    return 0;
}