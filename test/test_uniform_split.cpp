#include "ModelAnalyze/ModelAnalyzer.h"
using namespace std;

int main()
{
    ModelAnalyzer analyzer = ModelAnalyzer("vgg19");
    std::cout<<"init"<<std::endl;
    if(!analyzer.UniformSplit(5))
    {
        cout<<"bad aim"<<endl;
    }

    return 0;
}