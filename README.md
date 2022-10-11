# OnnxsInferenceManager

In order to eliminate the negative effects of fake-multi-threading mechanism of `Python` course by `GIL`, we eventually decided to refactor the code in `C++`. Raw Project with Python can still be found at: https://github.com/EdgeScheduler/OnnxSplitRunner

Develop Environment:
* gcc and g++ version 8.4.0
* onnxruntime-gpu for C++ version 1.12.1
* C++ compiler support:
  * g++ -std=c++17
  * `-lstdc++fs` support
  > example: g++ -std=c++17 main.cpp -o exec -lstdc++fs -lonnxruntime
* [nlohmann::json](https://github.com/nlohmann/json) library install.
* onnxruntime-gpu driver.

others:
* [C++ chinese manul](https://www.apiref.com/cpp-zh/cpp/filesystem/path.html)
* [onnxruntime C++ API](https://onnxruntime.ai/docs/api/c/namespace_ort.html#details)
* [nlohmann::json](https://github.com/nlohmann/json)

env: `10-CPU core, GTX 2080Ti`
