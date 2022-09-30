#include <limits.h>
#include <sys/time.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <algorithm>
#include <iostream>
#include <new>
#include <numeric>
#include <vector>

#include "AudioFile.h"

#define BASE_AUDIO_N (88)
#define SKIP_N (22)
#define MAX_LENGTH (7358334)

typedef struct {
    unsigned int sum;
    unsigned int id;
} min_sum;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

template <typename T_in, typename T_out>
struct AbsDiff {
    __host__ __device__ constexpr T_out operator()(const T_in &a, const T_in &b) const {
        return abs(a - b);
    }
};

template <typename T_in, typename T_out>
struct AbsPlus {
    __host__ __device__ constexpr T_out operator()(const T_in &a, const T_in &b) const {
        return a + abs(b);
    }
};

__global__ void diffSum(short *problem, short *src, unsigned int *sums, const int problemLen, const int sourceLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx * SKIP_N + 1;  // 1 start
    if (index >= (problemLen + sourceLen)) {
        return;
    }
    int clip_starti = max(0, index - sourceLen);
    int clip_endi = min(index, problemLen);
    int src_starti = max(sourceLen - index, 0);
    int src_endi = min(sourceLen, sourceLen + problemLen - index);

    unsigned int sum = 0;
    for (auto i = clip_starti, j = src_starti; i < clip_endi; i++, j++) {
        sum += abs(problem[i] - src[j]);
    }

    //残りの加算
    // for (auto i = 0; i < clip_starti; i++) {
    //     sum += abs(problem[i]);
    // }
    sum += thrust::reduce(thrust::cuda::par, problem, problem + clip_starti, 0, AbsPlus<int, unsigned int>());

    // for (auto i = clip_endi; i < problemLen; i++) {
    //     sum += abs(problem[i]);
    // }
    sum += thrust::reduce(thrust::cuda::par, problem + clip_endi, problem + problemLen, 0, AbsPlus<int, unsigned int>());

    sums[idx] = sum;
}

int main() {
    cudaStream_t streams[BASE_AUDIO_N];
    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        cudaStreamCreate(&streams[i]);
    }

    min_sum sums[BASE_AUDIO_N];
    double iStart = cpuSecond();
    // wave読み込み
    AudioFile<short> problem_wave("samples/original/problem4.wav");
    AudioFile<short> baseAudios[BASE_AUDIO_N];
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
    thrust::device_vector<short> baseAudios_d[BASE_AUDIO_N];
    thrust::device_vector<unsigned int> sum_tmp[BASE_AUDIO_N];
    problem_d = problem_wave.samples[0];

    // processing
    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        new (sum_tmp + i) thrust::device_vector<unsigned int>((problem_length + baseAudio_length[i] - 2) / SKIP_N);
        //転送
        new (baseAudios_d + i) thrust::device_vector<short>(baseAudio_length[i]);
        cudaMemcpyAsync(thrust::raw_pointer_cast(baseAudios_d[i].data()), baseAudios[i].samples[0].data(), baseAudios[i].samples[0].size(), cudaMemcpyHostToDevice, streams[i]);
    }
    cudaThreadSynchronize();

    dim3 block(256);
    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        dim3 grid(((problem_length + baseAudio_length[i] - 2) / SKIP_N + block.x - 1) / block.x);
        printf("baseAudio ID: %d, Block: %d, Grid: %d\n", i + 1, block.x, grid.x);
        printf("sums_d size: %d, %d\n", (problem_length + baseAudio_length[i] - 2) / SKIP_N, grid.x * block.x);
        diffSum<<<grid, block, 0, streams[i]>>>(thrust::raw_pointer_cast(problem_d.data()), thrust::raw_pointer_cast(baseAudios_d[i].data()), thrust::raw_pointer_cast(sum_tmp[i].data()), problem_length, baseAudio_length[i]);
    }

    cudaThreadSynchronize();

    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        sums[i].sum = thrust::reduce(thrust::device, sum_tmp[i].begin(), sum_tmp[i].end(), UINT_MAX, thrust::minimum<unsigned int>());
        sums[i].id = i;
    }

    std::sort(sums, sums + BASE_AUDIO_N, [](const min_sum &a, const min_sum &b) { return a.sum < b.sum; });
    printf("[");
    for (auto s : sums) {
        // printf("ID: %d SUM: %d\n", s.id, s.sum);
        printf("[%d, %d],", s.id + 1, s.sum);
    }
    printf("]\n");

    // post process
    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        baseAudios_d[i].~device_vector();
        cudaFree(&baseAudios_d[i]);  //配置newで確保したのにcudaFreeで解放するみたい
        sum_tmp[i].~device_vector();
        cudaFree(&sum_tmp[i]);

        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
