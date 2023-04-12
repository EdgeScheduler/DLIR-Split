#include "ModelAnalyze/ModelAnalyzer.h"
#include "cmdline.h"
#include "Common/JsonSerializer.h"
#include <numeric>
#include "Benchmark/evaluate_models.h"
using namespace std;

#define MAX_THREAD 20

int main(int argc, char *argv[])
{
    cmdline::parser parser;
    parser.set_program_name("DLIR-SPLIT");
    parser.add<int>("count", 'c', "how many count to split [>0]", false, 3);
    parser.add<std::string>("model", 'm', "model name", false, "vgg19");
    parser.add<int>("threads", 't', "how many threads allowed, if value<1, will auto-caculate", false, 0);
    parser.add<bool>("auto-end", 'e', "allow early end.", false, true);
    parser.add<bool>("enable-cache", 'h', "allow to use cache to speed bench.", false, false);
    parser.add<std::string>("GPU-Tag", 'g', "gpu tag.", false, "default");
    parser.add<int>("polution", 'p', "polution.", false, 20); // 30
    parser.add<int>("generation", 'r', "generations.", false, 100); // 100
    parser.add<int>("stall-max", 's', "how may gens to end if fit.", false, 7); // 7

    parser.parse_check(argc, argv);
    int count = parser.get<int>("count") > 0 ? parser.get<int>("count") : 3;
    std::string model_name=parser.get<std::string>("model");

// <<<<<<< Updated upstream
    int n_thread=parser.get<int>("threads");
    if(n_thread<1)
    {
        n_thread=(MAX_THREAD-2)/count;
    }

    cout<<"=> split "<<model_name<<" to "<<count<<" with threads="<<n_thread<<endl;
// =======
//     cout<<"=> split "<<model_name<<" to "<<count<<endl;

//     // 原代码
// >>>>>>> Stashed changes
    ModelAnalyzer analyzer = ModelAnalyzer(model_name);
    if(!analyzer.UniformSplit(count,parser.get<std::string>("GPU-Tag"),parser.get<bool>("enable-cache"),n_thread,parser.get<bool>("auto-end"),parser.get<int>("generation"),parser.get<int>("polution"), 0.01, parser.get<int>("stall-max")))
    {
        cout<<"bad aim"<<endl;
    }
    else
    {
        cout<<"run to end."<<endl;
    }

    //随机开销标准差
    // ModelAnalyzer analyzer = ModelAnalyzer(model_name);
    // std::vector<float> costs;
    // double sigma;
    // nlohmann::json result;
    // int cnt = 0;
    // for(int i = 1; i < analyzer.size() - 1; i+=3)
    // {
    //     for(int j = i + 1; j < analyzer.size(); j+=3)
    //     {
            
    //         sigma = analyzer.SplitAndEvaluateChilds(costs, {analyzer[i], analyzer[j]});
    //         result[std::to_string(cnt)]["breakpoint1"] = i;
    //         result[std::to_string(cnt)]["breakpoint2"] = j;
    //         result[std::to_string(cnt)]["sigma"] = sigma;
    //         result[std::to_string(cnt)]["cost"] = accumulate(costs.begin(), costs.end(), 0);
    //         cnt ++;
    //         std::cout<<"end"<<std::endl;
    //     }
    // }
    
    // JsonSerializer::StoreJson(result, "vgg19.json");


    //以算子为单位切分
    // ModelAnalyzer analyzer = ModelAnalyzer(model_name);
    // double sigma;
    // nlohmann::json result;
    // // analyzer.SplitAndStoreChilds(analyzer.GetAllNodes());
    // analyzer.SplitAndStoreChilds({analyzer[10], analyzer[19]});
    // nlohmann::json time_evaluate_dict = evam::TimeEvaluateChildModels(model_name);
    // std::vector<double> costs;
    // for(int i = 0; i < analyzer.size(); i++)
    // {
    //     costs.emplace_back(time_evaluate_dict["childs"][to_string(i)]["avg"].get<double>());
    // }


    // double avg = accumulate(costs.begin(), costs.end(), 0) / 3;
    // int size = costs.size();
    // double min = 1000;
    // double tmp;
    // int p1; int p2;
    // for(int i = 1; i < size - 1; i++)
    // {
    //     for(int j = i + 1; j < size; j++)
    //     {
    //         tmp = sqrt((pow((accumulate(costs.begin(), costs.begin() + i, 0)) - avg, 2) + pow((accumulate(costs.begin()+i, costs.begin()+j, 0))-avg, 2) + 
    //         pow((accumulate(costs.begin()+j, costs.end(), 0))-avg, 2)) / 3);
    //         if(tmp < min)
    //         {
    //             min = tmp;
    //             p1 = i;
    //             p2 = j;
    //         }
                
    //     }
    // }
    // std::cout<<"p1: "<<p1<<std::endl<<"p2: "<<p2<<std::endl<<"min: "<<min<<std::endl;
    // std::cout<<"part1: "<<accumulate(costs.begin(), costs.begin() + p1, 0)<<std::endl<<"part2: "<< accumulate(costs.begin()+p1, costs.begin()+p2, 0)<<std::endl<<"part3: "<<
    // accumulate(costs.begin()+p2, costs.end(), 0)<<std::endl;





    return 0;
}