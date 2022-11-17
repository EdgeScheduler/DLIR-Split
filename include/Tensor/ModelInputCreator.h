#ifndef __MODELINPUTCREATOR_H__
#define __MODELINPUTCREATOR_H__

#include <memory>
#include "ModelTensorsInfo.h"
#include "TensorValue.hpp"

class ModelInputCreator
{
public:
    ModelInputCreator(const TensorsInfo &tensorsInfo);

    /// @brief create datas for model with map, return share_ptr
    /// @return
    std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>> CreateInput();

private:
    std::map<std::string, TensorValue<float>> dataTemplate;
    TensorsInfo tensorsInfo;
};

#endif // __MODELINPUTCREATOR_H__: tensorsInfo(tensorsInfo){}