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

#define BASE_AUDIO_N (88)

typedef struct {
    unsigned int sum;
    unsigned int id;
} min_sum;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void diffSum(short *problem, short *src, unsigned int *sums, const int problemLen, const int sourceLen) {
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

        //残りの加算
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
    min_sum sums[BASE_AUDIO_N];
    double iStart = cpuSecond();
    // wave読み込み
    AudioFile<short> problem_wave("samples/original/problem4.wav");
    thrust::host_vector<short> problem_h(problem_wave.samples[0].begin(), problem_wave.samples[0].end());
    AudioFile<short> baseAudios[BASE_AUDIO_N];
    thrust::host_vector<short> baseAudios_h[BASE_AUDIO_N];
    int baseAudio_length[BASE_AUDIO_N];

    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        char buf[1000];
        snprintf(buf, sizeof(buf), "samples/JKspeech/%d.wav", i + 1);

        baseAudios[i].load(buf);
        baseAudio_length[i] = baseAudios[i].getNumSamplesPerChannel();
    }
    printf("%f[s] needed to read problems and baseAudios\n", cpuSecond() - iStart);

    problem_wave.printSummary();

    // デバイス（GPU）側メモリに領域を確保
    int problem_length = problem_wave.getNumSamplesPerChannel();
    thrust::device_vector<short> problem_d(problem_length);
    problem_d = problem_h;

    // processing
    dim3 block(512);
    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        thrust::device_vector<unsigned int> sums_d(problem_length + baseAudio_length[i] - 2);
        thrust::device_vector<short> baseAudio_d(baseAudio_length[i]);
        baseAudio_d = baseAudios[i].samples[0];

        dim3 grid((problem_length + baseAudio_length[i] + block.x - 1 - 1) / block.x);
        printf("baseAudio ID: %d, Block: %d, Grid: %d\n", i + 1, block.x, grid.x);
        diffSum<<<grid, block>>>(thrust::raw_pointer_cast(problem_d.data()), thrust::raw_pointer_cast(baseAudio_d.data()), thrust::raw_pointer_cast(sums_d.data()), problem_length, baseAudio_length[i]);
        auto iter = thrust::min_element(sums_d.begin(), sums_d.end());
        sums[i].sum = *iter;
        sums[i].id = i;
    }

    for (auto s : sums) {
        printf("ID: %d SUM: %d\n", s.id, s.sum);
    }

    return 0;
}