#ifndef TENSORVALUE_H
#define TENSORVALUE_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <ostream>
#include <memory>
#include <random>
#include <vector>
#include <iostream>
#include "ValueInfo.h"

/// @brief object to store tensor-info and datas
template <class T = float>
class TensorValue
{
public:
    /// @brief Init class and create random value.
    /// @param valueInfo shape and name Info
    /// @param allocator ort allocator
    /// @param memType memory type
    TensorValue(const ValueInfo &valueInfo, bool initByRandom = false, OrtAllocatorType allocator = OrtAllocatorType::OrtArenaAllocator, OrtMemType memType = OrtMemType::OrtMemTypeDefault);

    /// @brief create random datas
    void Random(int min = 0, int max = 1);

    /// @brief set label for tensor-info
    /// @param label usually label-name in model.
    void SetLabel(std::string label = "");

    /// @brief get raw data
    /// @return std::vector<T> object
    std::vector<T> GetData() const;

    /// @brief to  Ort::Value
    /// @return
    Ort::Value &&ToTensor() const;

    /// @brief get tensor-info
    /// @return ValueInfo Info
    ValueInfo GetValueInfo() const;

    /// @brief print Tensor-info
    // template<typename ValueType=float>
    void Print(int64_t max_length = 30, bool print_tensor_info = true) const;

    TensorValue<T> &operator=(Ort::Value &&value);

private:
    ValueInfo valueInfo;
    std::vector<T> data;
    OrtAllocatorType allocator;
    OrtMemType memType;
};

template <class T>
TensorValue<T>::TensorValue(const ValueInfo &valueInfo, bool initByRandom, OrtAllocatorType allocator, OrtMemType memType) : data(valueInfo.GetDataCount())
{
    this->valueInfo = valueInfo;
    this->allocator = allocator;
    this->memType = memType;

    if (initByRandom)
    {
        this->Random();
    }
}

template <class T>
void TensorValue<T>::Random(int min, int max)
{
    static std::default_random_engine engin(time(NULL));
    std::uniform_real_distribution<double> uniform_creator(min, max);

    for (int i = 0; i < this->data.size(); i++)
    {
        this->data[i] = (T)uniform_creator(engin);
    }
}

template <class T>
TensorValue<T> &TensorValue<T>::operator=(Ort::Value &&value)
{
    std::memcpy(this->data.data(), value.GetTensorMutableData<T>(), sizeof(T) * this->valueInfo.GetDataCount());
    return *this;
}

template <class T>
Ort::Value &&TensorValue<T>::ToTensor() const
{
    return Ort::Value::CreateTensor<T>(Ort::MemoryInfo::CreateCpu(allocator, memType), data.data(), data.size(), valueInfo.GetShape().data(), valueInfo.GetDimSize());
}

template <class T>
void TensorValue<T>::SetLabel(std::string label)
{
    this->valueInfo.SetName(label);
}

template <class T>
std::vector<T> TensorValue<T>::GetData() const
{
    return this->data;
}

template <class T>
ValueInfo TensorValue<T>::GetValueInfo() const
{
    return this->valueInfo;
}

template <class T>
void TensorValue<T>::Print(int64_t max_length, bool print_tensor_info) const
{
    if (print_tensor_info)
    {
        std::cout << this->valueInfo << std::endl;
    }

    int64_t length = max_length <= this->valueInfo.GetDataCount() ? max_length : this->valueInfo.GetDataCount();
    std::cout << "[";
    for (int64_t i = 0; i < length; i++)
    {
        if (i == 0)
        {
            std::cout << this->data[i];
        }
        else
        {
            std::cout << ", " << this->data[i];
        }
    }

    if (length < this->valueInfo.GetDataCount())
    {
        std::cout << " ...]"<<std::endl;
    }
    else
    {
        std::cout << "]" << std::endl;
    }
}

#endif // !TENSORVALUE_H
