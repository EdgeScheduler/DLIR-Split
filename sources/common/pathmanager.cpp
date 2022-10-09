#include"../../headers/common/pathmanager.h"

#include<stdio.h>
#include<unistd.h>
#include<string.h>

// calculate project path
std::filesystem::path PathManager::projectRootFold=[]()->std::filesystem::path{
    std::filesystem::path p;

    char execPath[1024];
    memset(execPath, 0, sizeof(execPath));
    if(readlink("/proc/self/exe", execPath, sizeof(execPath)-1)>=1023)      // read exec path
    {
        std::cout<<"warning: process exec path is too long(>=1023), we try to use work path instead, may meet some error."<<std::endl;
        char *workPath=getcwd(NULL,0);
        if(workPath==NULL)
        {
            p="./";
        }
        else
        {
            p=workPath;
        }
    }
    else
    {
        p=execPath;
        p=p.parent_path();
    }

    if(p.parent_path().filename()=="bin" && (p.filename()=="release" || p.filename()=="debug"))
    {
        p=p.parent_path().parent_path();
    }
    return p.make_preferred();
}();

std::filesystem::path PathManager::GetProjectRootFold()
{
    return PathManager::projectRootFold;
}