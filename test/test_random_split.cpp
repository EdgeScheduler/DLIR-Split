#include <filesystem>
#include <iostream>
#include<algorithm>
#include "ModelAnalyze/ModelAnalyzer.h"
#include "onnx/shape_inference/implementation.h"
#include "Utils/helper.h"
#include "Benchmark/evaluate_models.h"
#include "Utils/UniformOptimizer.h"
using namespace std;

std::vector<int> generate_indexs(int max, int count)
{
    std::vector<int> indexs;
    indexs.push_back(0);

    while(indexs.size()<count)
    {
        int value=rand()%max;
        while(std::count(indexs.begin(), indexs.end(), value)>0)
        {
            value=rand()%max;
        }

        indexs.push_back(value);
    }

    sort(indexs.begin(), indexs.end());

    return indexs;
}

int main()
{
    ModelAnalyzer analyzer = ModelAnalyzer("vgg19");

    auto &nodes=analyzer.GetAllNodes();
    int model_count=3;

    
    // // test same count
    // cout<<"same childs count"<<endl;
    // for(int i=0;i<5;i++)
    // {
    //     cout<<endl<<i<<":"<<endl;
    //     std::vector<int> indexs=generate_indexs(nodes.size(),model_count);
    //     std::vector<GraphNode> split_nodes;
    //     for(int index: indexs)
    //     {
    //         split_nodes.push_back(nodes[index]);
    //     }
    //     analyzer.SplitAndStoreChilds(split_nodes);
    //     evam::TimeEvaluateChildModels("vgg19", "GPU-2080Ti", false);
    // }

    cout<<"adding childs count"<<endl;

    std::vector<int> indexs=generate_indexs(nodes.size(),5);
    for(int i=1;i<=5;i++)
    {
        cout<<endl<<i<<":"<<endl;
        std::vector<GraphNode> split_nodes;
        for(int j=0;j<i;j++)
        {
            split_nodes.push_back(nodes[indexs[j]]);
        }

        analyzer.SplitAndStoreChilds(split_nodes);
        evam::TimeEvaluateChildModels("vgg19", "GPU-2080Ti", false);
    }
    
    
   cout<<endl;
    


    return 0;
}