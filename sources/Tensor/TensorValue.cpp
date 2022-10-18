// #include "../../include/Tensor/TensorValue.h"

// #include <random>
// #include <vector>
// #include <iostream>

// template <class T>
// TensorValue<T>::TensorValue(const ValueInfo &valueInfo, bool initByRandom, OrtAllocatorType allocator, OrtMemType memType) : data(valueInfo.GetDataCount())
// {
//     this->valueInfo = valueInfo;
//     this->allocator = allocator;
//     this->memType = memType;

//     if (initByRandom)
//     {
//         this->Random();
//     }
// }

// template <class T>
// void TensorValue<T>::Random(int min, int max)
// {
//     std::default_random_engine engin;
//     std::uniform_real_distribution<double> uniform_creator(min, max);

//     for (int i = 0; i < this->data.size(); i++)
//     {
//         this->data[i] = (T)uniform_creator(engin);
//     }
// }

// template <class T>
// TensorValue<T> &TensorValue<T>::operator=(Ort::Value &&value)
// {
//     std::memcpy(this->data.data(), value.GetTensorMutableData<T>(), sizeof(T) * this->valueInfo.GetDataCount());
//     return *this;
// }

// template <class T>
// Ort::Value &&TensorValue<T>::ToTensor() const
// {
//     return Ort::Value::CreateTensor<T>(Ort::MemoryInfo::CreateCpu(allocator, memType), data.data(), data.size(), valueInfo.GetShape().data(), valueInfo.GetDimSize());
// }

// template <class T>
// void TensorValue<T>::SetLabel(std::string label)
// {
//     this->valueInfo.SetName(label);
// }

// template <class T>
// std::vector<T> TensorValue<T>::GetData() const
// {
//     return this->data;
// }

// template <class T>
// ValueInfo TensorValue<T>::GetValueInfo() const
// {
//     return this->valueInfo;
// }

// template <class T>
// void TensorValue<T>::Print(int64_t max_length, bool print_tensor_info) const
// {
//     if (print_tensor_info)
//     {
//         std::cout << this->valueInfo << std::endl;
//     }

//     int64_t length = max_length <= this->valueInfo.GetDataCount() ? max_length : this->valueInfo.GetDataCount();
//     std::cout << "[";
//     for (int64_t i = 0; i < length; i++)
//     {
//         if (i == 0)
//         {
//             std::cout << this->data[i];
//         }
//         else
//         {
//             std::cout << ", " << this->data[i];
//         }
//     }

//     if (length < this->valueInfo.GetDataCount())
//     {
//         std::cout << " ...]";
//     }
//     else
//     {
//         std::cout << "]" << std::endl;
//     }
// }
