#include "ModelAnalyze/ModelAnalyzer.h"
using namespace std;

int main()
{
    ModelAnalyzer analyzer = ModelAnalyzer("vgg19");
    if(!analyzer.UniformSplit(3))
    {
        cout<<"bad aim"<<endl;
    }

    return 0;
}