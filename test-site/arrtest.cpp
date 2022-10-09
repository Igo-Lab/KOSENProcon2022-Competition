
/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <sys/time.h>

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;
using comb = pair<uint32_t, uint32_t>;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void arrsort(uint32_t **arr, int len) {
    sort(
        (comb *)arr,
        (comb *)arr + len,
        [](const comb &a, const comb &b) { return a.second < b.second; });
}

int main() {
    uint32_t arr[2 << 15][2];
    for (auto i = 0; i < 2 << 15; i++) {
        arr[i][0] = i;
        arr[i][1] = rand();
    }

    cout << "go" << endl;
    double istart = cpuSecond();
    arrsort((uint32_t **)arr, 2 << 15);
    printf("%lf\n", cpuSecond() - istart);

    vector<vector<uint32_t>> arr2(2 << 15, vector<uint32_t>(2));
    for (auto i = 0; i < 2 << 15; i++) {
        arr[i][0] = i;
        arr[i][1] = rand();
    }
    istart = cpuSecond();
    sort(
        arr2.begin(),
        arr2.end(),
        [](const auto &a, const auto &b) { return a[1] < b[1]; });
    printf("%lf\n", cpuSecond() - istart);

    cout << "Hello World";

    return 0;
}
