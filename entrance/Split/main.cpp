#include "ModelAnalyze/ModelAnalyzer.h"
#include "cmdline.h"
using namespace std;

int main(int argc, char *argv[])
{
    cmdline::parser parser;
    parser.set_program_name("DLIR-SPLIT");
    parser.add<int>("count", 'c', "how many count to split [>0]", false, 3);
    parser.add<std::string>("model", 'm', "model name", false, "vgg19");
    parser.add<int>("threads", 't', "how many threads allowed, if value<1, default(1) will be valid.", false, 1);
    parser.add<bool>("autoend", 'e', "allow early end.", false, true);

    parser.parse_check(argc, argv);
    int count = parser.get<int>("count") > 0 ? parser.get<int>("count") : 3;
    std::string model_name=parser.get<std::string>("model");

    cout<<"=> split "<<model_name<<" to "<<count<<endl;
    ModelAnalyzer analyzer = ModelAnalyzer(model_name);
    if(!analyzer.UniformSplit(count,"RTX-2080Ti",parser.get<int>("threads"),parser.get<bool>("autoend"),100,50, 0.01, 5))
    {
        cout<<"bad aim"<<endl;
    }
    else
    {
        cout<<"run to end."<<endl;
    }

    return 0;
}