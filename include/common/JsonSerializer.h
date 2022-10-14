#ifndef __JSONSERIALIZER_H__
#define __JSONSERIALIZER_H__

#include <nlohmann/json.hpp>
#include <filesystem>

namespace JsonSerializer
{
    /// @brief load json from path, if error happen, can be check by obj.is_discarded(), value ok will return false, else true.
    /// @param path where to find json
    /// @return nlohmann::json object.
    nlohmann::json LoadJson(const std::filesystem::path &path);

    /// @brief store nlohmann::json object to path
    /// @param json nlohmann::json object
    /// @param path where to save json.
    /// @return success or not
    bool StoreJson(const nlohmann::json &json, const std::filesystem::path &path, bool enable_null_json = false);
}

#endif // __JSONSERIALIZER_H__