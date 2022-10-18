#ifndef __MODELINPUTCREATOR_H__
#define __MODELINPUTCREATOR_H__

#include "ModelTensorsInfo.h"
#include "TensorValue.hpp"

class ModelInputCreator
{
public:
    ModelInputCreator(const TensorsInfo& tensorsInfo);
    std::map<std::string, TensorValue<float>> CreateInput();

private:
    std::map<std::string, TensorValue<float>> dataTemplate;
    TensorsInfo tensorsInfo;
};

#endif // __MODELINPUTCREATOR_H__: tensorsInfo(tensorsInfo){}