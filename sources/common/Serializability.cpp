#include "../../include/common/Serializability.h"
#include "../../include/common/JsonSerializer.h"

void Serializability::LoadFromJsonFile(const std::filesystem::path &path)
{
    this->LoadFromJson(JsonSerializer::LoadJson(path));
}

bool Serializability::StoreJsonWithPath(const std::filesystem::path &path) const
{
    return JsonSerializer::StoreJson(this->ToJson(), path, true);
}
