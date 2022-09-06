/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <iostream>
#include <algorithm>

using namespace std;
using comp_pair = std::pair<uint32_t, uint32_t>;

void test(uint32_t **a){
    std::sort((comp_pair *)a, (comp_pair *)a + 3, [](const auto &a, const auto &b) { return a.second < b.second; });
    for(auto i = 0;i < 3;i++){
        printf("%d %d\n",a[i][0],a[i][1]);
    }
}

int main()
{
    uint32_t a[][2] = {{1,2},{2,100},{3,300}};
    std::sort((comp_pair *)a, (comp_pair *)a + 3, [](const auto &a, const auto &b) { return a.second < b.second; });
    for(auto i = 0;i < 3;i++){
        printf("%d %d\n",a[i][0],a[i][1]);
    }
    return 0;
}
