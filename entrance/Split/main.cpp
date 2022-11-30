#include "ModelAnalyze/ModelAnalyzer.h"
#include "cmdline.h"
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
    parser.add<int>("polution", 'p', "polution.", false, 50);
    parser.add<int>("generation", 'r', "generations.", false, 100);
    parser.add<int>("stall-max", 's', "how may gens to end if fit.", false, 3);

    parser.parse_check(argc, argv);
    int count = parser.get<int>("count") > 0 ? parser.get<int>("count") : 3;
    std::string model_name=parser.get<std::string>("model");

    int n_thread=parser.get<int>("threads");
    if(n_thread<1)
    {
        n_thread=(MAX_THREAD-2)/count;
    }

    cout<<"=> split "<<model_name<<" to "<<count<<" with threads="<<n_thread<<endl;
    ModelAnalyzer analyzer = ModelAnalyzer(model_name);
    if(!analyzer.UniformSplit(count,parser.get<std::string>("GPU-Tag"),parser.get<bool>("enable-cache"),n_thread,parser.get<bool>("auto-end"),parser.get<int>("generation"),parser.get<int>("polution"), 0.01, parser.get<int>("stall-max")))
    {
        cout<<"bad aim"<<endl;
    }
    else
    {
        cout<<"run to end."<<endl;
    }

    return 0;
}