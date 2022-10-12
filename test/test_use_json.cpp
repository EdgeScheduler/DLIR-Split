#include <nlohmann/json.hpp>
#include <iostream>
using namespace std;

int main()
{
    nlohmann::json config_json = nlohmann::json::parse(R"({"obj":null})"); //构建json对象

    nlohmann::json new_json=config_json["obj"];
    new_json["obj"]={1,2,3};

    auto a=new_json["obj"];
    std::vector<int> b=a.get<std::vector<int>>();

    for(auto i=b.begin();i<b.end();i++)
    {

        cout<<*i<<endl;
    }

    cout<<new_json.contains("111")<<new_json.contains("obj")<<endl;

    return 0;
}