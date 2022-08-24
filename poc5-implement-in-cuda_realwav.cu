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

#include "AudioFile.h"

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
    // wave読み込み
    AudioFile<short> problem_wave(
        "/content/drive/MyDrive/Colab Notebooks/samples/original/problem4.wav");
    AudioFile<short> srcJ01_wave(
        "/content/drive/MyDrive/Colab Notebooks/samples/JKspeech/J01.wav");

    int problem_length = problem_wave.getNumSamplesPerChannel();
    int srcJ01_length = srcJ01_wave.getNumSamplesPerChannel();

    problem_wave.printSummary();
    // ホスト（CPU）側メモリに領域を確保
    double iStart = cpuSecond();
    thrust::host_vector<short> problem_h(problem_wave.samples[0].begin(),
                                         problem_wave.samples[0].end());
    thrust::host_vector<short> src_h(srcJ01_wave.samples[0].begin(),
                                     srcJ01_wave.samples[0].end());
    thrust::host_vector<unsigned int> sums_h(problem_length + srcJ01_length -
                                             2);

    printf("%f[s] needed to gen arrs\n", cpuSecond() - iStart);

    // デバイス（GPU）側メモリに領域を確保
    thrust::device_vector<short> problem_d(problem_length);
    thrust::device_vector<short> src_d(srcJ01_length);
    thrust::device_vector<unsigned int> sums_d(problem_length + srcJ01_length -
                                               2);
    problem_d = problem_h;
    src_d = src_h;

    std::cout << "Copied to Device." << std::endl;

    dim3 block(512);
    dim3 grid((problem_length + srcJ01_length + block.x - 1 - 1) / block.x);
    printf("Block: %d, Grid: %d\n", block.x, grid.x);
    diffSum<<<grid, block>>>(thrust::raw_pointer_cast(problem_d.data()),
                             thrust::raw_pointer_cast(src_d.data()),
                             thrust::raw_pointer_cast(sums_d.data()),
                             problem_length, srcJ01_length);

    auto iter = thrust::min_element(sums_d.begin(), sums_d.end());
    unsigned int pos = iter - sums_d.begin();
    std::cout << "MinVal: " << *iter << " pos: " << pos << std::endl;

    return 0;
}