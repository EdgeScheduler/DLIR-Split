#include <nlohmann/json.hpp>
#include <iostream>
using namespace std;

int main()
{
    nlohmann::json config_json = nlohmann::json::parse(R"({})"); //构建json对象

    nlohmann::json new_json;
    new_json["obj"]=config_json;


    cout << new_json << endl; //输出json对象值
    return 0;
}