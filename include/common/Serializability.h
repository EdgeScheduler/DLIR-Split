#ifndef __SERIALIZABILITY_H__
#define __SERIALIZABILITY_H__

#include <nlohmann/json.hpp>

/// @brief serialize object to nlohmann::json
class Serializability
{
public:
    /// @brief serial to json object
    /// @return one nlohmann::json object.
    virtual nlohmann::json ToJson() const = 0;

    /// @brief load class to json
    /// @param nlohmann::json  object
    virtual void LoadFromJson(const nlohmann::json &json) = 0;

    /// @brief load object from path, if error happen, can get empty class.
    /// @param path where to find json
    /// @return nlohmann::json object.
    virtual void LoadFromJsonFile(const std::filesystem::path &path);

    /// @brief store current object to path
    /// @param path where to save json.
    /// @return success or not
    virtual bool StoreJsonWithPath(const std::filesystem::path &path) const;
};

#endif // __SERIALIZABILITY_H__