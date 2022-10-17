#include<iostream>
using namespace std;

int main()
{
    int f=12345;
    int m=0;
    {
        int& a=f;
        m=a;
        cout<<a<<endl;
        a++;
    }
    cout<<m<<endl;
    f++;

    cout<<m<<endl;
    return 0;
}