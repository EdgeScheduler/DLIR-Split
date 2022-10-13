#include<iostream>

using namespace std;

int main()
{
    int&& a=10;
    int& p=a;
    cout<<p<<a<<endl;
    return 0;
}