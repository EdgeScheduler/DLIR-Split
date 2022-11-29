#include "ModelAnalyze/ModelAnalyzer.h"
using namespace std;

int main()
{
    ModelAnalyzer analyzer = ModelAnalyzer("vgg19");
    if(!analyzer.UniformSplit(3,"RTX-2080Ti",true,false,100,50))
    {
        cout<<"bad aim"<<endl;
    }

    return 0;
}