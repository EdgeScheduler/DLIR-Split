#ifndef PATHManager_H
#define PATHManager_H

#include<iostream>
#include<filesystem>

class PathManager
{
public:
    static std::filesystem::path GetProjectRootFold();

private:
    static std::filesystem::path projectRootFold;
};




#endif // !PATHManager_H