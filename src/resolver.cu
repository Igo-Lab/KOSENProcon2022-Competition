
#include <cuda_runtime.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <iostream>

#include "resolver.h"

constexpr size_t BASE_AUDIO_N = 88;
constexpr size_t BLOCK_N = 256;
using comp_pair = std::pair<uint32_t, uint32_t>;

int16_t *srcAudios[BASE_AUDIO_N];  // gpuのメモリポインタ．マルチスレッドで呼び出すときに問題になりそう．
uint32_t *sum_tmp[BASE_AUDIO_N];
int32_t srclens[BASE_AUDIO_N];
cudaStream_t streams[BASE_AUDIO_N];
bool isInit = false;

__global__ void diffSum(const int16_t *__restrict__ chunk, const int16_t *__restrict__ src, unsigned int *sums, const int chunkLen, const int sourceLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx + 1;  // 1 start
    if (index >= (chunkLen + sourceLen)) {
        return;
    }
    int clip_starti = max(0, index - sourceLen);
    int clip_endi = min(index, chunkLen);
    int src_starti = max(sourceLen - index, 0);
    int src_endi = min(sourceLen, sourceLen + chunkLen - index);

    unsigned int sum = 0;

    //生problemの前加算
#pragma unroll 8
    for (auto i = 0; i < clip_starti; i++) {
        sum += abs(chunk[i]);
    }

#pragma unroll 8
    for (auto i = clip_starti, j = src_starti; i < clip_endi; i++, j++) {
        sum += abs(chunk[i] - src[j]);
        // sum = __sad(chunk[i], src[j], sum);
    }

    //生problemの後加算
#pragma unroll 8
    for (auto i = clip_endi; i < chunkLen; i++) {
        sum += abs(chunk[i]);
    }

    sums[idx] = sum;
}

__attribute__((constructor)) void initgpu() {
    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        cudaStreamCreate(&streams[i]);
    }

    isInit = true;
}

//とりあえず何も考えずsrcをコピーして解答領域は都度都度確保することに
// TODO:動確したら直す
void memcpy_src2gpu(const int16_t **srcs, const int32_t *lens) {
    if (!isInit) {
        std::cout << "Didn't be inited. Not processing." << std::endl;
        return;
    }

    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        //メモリ確保
        cudaMallocAsync((void **)&srcAudios[i], sizeof(int16_t) * lens[i], streams[i]);
        // 読みデータのコピー
        cudaMemcpyAsync(srcAudios[i], srcs[i], sizeof(int16_t) * lens[i], cudaMemcpyHostToDevice, streams[i]);
        //読みデータ配列の長さを記録しておく
        srclens[i] = lens[i];
    }
    cudaDeviceSynchronize();
}

void resolver(const int16_t *chunk, const int32_t chunk_len, uint32_t **result) {
    int16_t *chunk_d;
    dim3 block(BLOCK_N);

    if (!isInit) {
        std::cout << "Didn't be initialized. Not processing." << std::endl;
        return;
    }

    cudaMalloc((void **)&chunk_d, sizeof(int16_t) * chunk_len);
    cudaMemcpy(chunk_d, chunk, sizeof(int16_t) * chunk_len, cudaMemcpyHostToDevice);

    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        //解答保存領域の確保
        cudaMallocAsync((void **)&sum_tmp[i], sizeof(uint32_t) * (chunk_len + srclens[i] - 2), streams[i]);

        dim3 grid(((chunk_len + srclens[i] - 2) + block.x - 1) / block.x);
        diffSum<<<grid, block, 0, streams[i]>>>(chunk_d, srcAudios[i], sum_tmp[i], chunk_len, srclens[i]);
    }

    cudaDeviceSynchronize();

    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        result[i][0] = i + 1;
        result[i][1] = thrust::reduce(
            thrust::device,
            sum_tmp[i],
            sum_tmp[i] + (chunk_len + srclens[i] - 2),
            UINT32_MAX,
            thrust::minimum<uint32_t>());
        cudaFreeAsync(sum_tmp[i], streams[i]);
    }

    std::sort((comp_pair *)result, (comp_pair *)result + BASE_AUDIO_N, [](const auto &a, const auto &b) { return a.second < b.second; });

    cudaFree(chunk_d);
    cudaDeviceSynchronize();
}

__attribute__((destructor)) void deinitgpu() {
    if (!isInit) return;
    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        cudaStreamDestroy(streams[i]);
    }
    isInit = false;
}