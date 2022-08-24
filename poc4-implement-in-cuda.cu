#include <sys/time.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#define PRB_LEN 5
#define SRC_LEN 10

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void diffSum(short *problem, short *src, unsigned int *sums,
                        const int problemLen, const int sourceLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx + 1;  // 1 start
    if (index < (problemLen + sourceLen)) {
        int clip_starti = max(0, index - sourceLen);
        int clip_endi = min(index, problemLen);
        int src_starti = max(sourceLen - index, 0);
        int src_endi = min(sourceLen, sourceLen + problemLen - index);

        unsigned int sum = 0;
        for (auto i = clip_starti, j = src_starti; i < clip_endi; i++, j++) {
            sum += abs(problem[i] - src[j]);
        }

        for (auto i = 0; i < clip_starti; i++) {
            sum += abs(problem[i]);
        }

        for (auto i = clip_endi; i < problemLen; i++) {
            sum += abs(problem[i]);
        }

        sums[idx] = sum;
    }
}

int main() {
    // ホスト（CPU）側メモリに領域を確保
    double iStart = cpuSecond();
    thrust::host_vector<short> problem_h(PRB_LEN);
    thrust::host_vector<short> src_h(SRC_LEN);
    thrust::host_vector<unsigned int> sums_h(PRB_LEN + SRC_LEN - 1);
    std::iota(problem_h.begin(), problem_h.end(), 1);
    std::iota(src_h.begin(), src_h.end(), 1);

    printf("%f[s] needed to gen arrs\n", cpuSecond() - iStart);

    std::cout << "problem array\n";
    std::for_each(problem_h.begin(), problem_h.end(),
                  [&](auto i) { std::cout << i << " "; });
    std::cout << std::endl;
    std::cout << "src array\n";
    std::for_each(src_h.begin(), src_h.end(),
                  [&](auto i) { std::cout << i << " "; });
    std::cout << std::endl;

    // デバイス（GPU）側メモリに領域を確保
    thrust::device_vector<short> problem_d(PRB_LEN);
    thrust::device_vector<short> src_d(SRC_LEN);
    thrust::device_vector<unsigned int> sums_d(PRB_LEN + SRC_LEN - 1);
    problem_d = problem_h;
    src_d = src_h;

    std::cout << "Copied to Device." << std::endl;

    dim3 grid(1);
    dim3 block(32);
    diffSum<<<grid, block>>>(thrust::raw_pointer_cast(problem_d.data()),
                             thrust::raw_pointer_cast(src_d.data()),
                             thrust::raw_pointer_cast(sums_d.data()), PRB_LEN,
                             SRC_LEN);

    sums_h = sums_d;
    std::for_each(sums_h.begin(), sums_h.end(),
                  [&](auto i) { std::cout << i << " "; });
    std::cout << std::endl;
    // thrust::device_vector<int>::iterator iter =
    //     thrust::max_element(device_input_first.begin(),
    //     device_input_first.end());

    // unsigned int pos = iter - device_input_first.begin();
    // int max_val = *iter;

    // std::cout << "MaxVal: " << max_val << " at " << pos << std::endl;
    return 0;
}