
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <iostream>

#include "resolver.h"

#define CUDA_SAFE_CALL(func)                                                                                                  \
    do {                                                                                                                      \
        cudaError_t err = (func);                                                                                             \
        if (err != cudaSuccess) {                                                                                             \
            fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(err);                                                                                                        \
        }                                                                                                                     \
    } while (0)

constexpr size_t BASE_AUDIO_N = 1;
constexpr size_t BLOCK_N = 256;
using comp_pair = std::pair<uint32_t, uint32_t>;

int16_t *srcAudios[BASE_AUDIO_N];  // gpuのメモリポインタ．マルチスレッドで呼び出すときに問題になりそう．
uint32_t srclens[BASE_AUDIO_N];
cudaStream_t streams[BASE_AUDIO_N];
bool isInit = false;
bool srcLoaded = false;

__global__ void diffSum(const int16_t *__restrict__ chunk, const int16_t *__restrict__ src, uint32_t *sums, const uint32_t chunkLen, const uint32_t sourceLen) {
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

//とりあえず何も考えずsrcをコピーして解答領域は都度都度確保することに
// TODO:動確したら直す
void memcpy_src2gpu(const int16_t **srcs, const uint32_t *lens) {
    if (!isInit) {
        std::cout << "Didn't be inited. Not processing." << std::endl;
        return;
    }

    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        //メモリ確保
        CUDA_SAFE_CALL(cudaMallocAsync((void **)&(srcAudios[i]), sizeof(int16_t) * lens[i], streams[i]));
        // 読みデータのコピー
        CUDA_SAFE_CALL(cudaMemcpyAsync(srcAudios[i], srcs[i], sizeof(int16_t) * lens[i], cudaMemcpyHostToDevice, streams[i]));

        srclens[i] = lens[i];
    }
    cudaDeviceSynchronize();

    // {
    //     // debug
    //     int16_t *tmparr;
    //     tmparr = (int16_t *)malloc(sizeof(int16_t) * lens[0]);
    //     cudaMemcpy(tmparr, srcAudios[0], sizeof(int16_t) * lens[0], cudaMemcpyDeviceToHost);
    //     for (auto i = 0; i < lens[0]; i++) {
    //         if (tmparr[i] != srcs[0][i]) {
    //             std::cout << "Error" << std::endl;
    //         }
    //     }
    //     std::cout << "pass" << std::endl;
    //     free(tmparr);
    // }
    srcLoaded = true;
}

// 元読みデータはindex-0スタート．つまり0～87
void resolver(const int16_t *chunk, const uint32_t chunk_len, const bool *mask, uint32_t **result) {
    int16_t *chunk_d;
    thrust::device_vector<uint32_t> sum_tmp[BASE_AUDIO_N];
    dim3 block(BLOCK_N);

    if (!isInit) {
        std::cout << "Didn't be initialized. Not processing." << std::endl;
        return;
    }

    CUDA_SAFE_CALL(cudaMalloc((void **)&chunk_d, sizeof(int16_t) * chunk_len));
    CUDA_SAFE_CALL(cudaMemcpy(chunk_d, chunk, sizeof(int16_t) * chunk_len, cudaMemcpyHostToDevice));

    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        if (mask[i]) {
            continue;
        }
        sum_tmp[i].resize((chunk_len + srclens[i] - 2));
    }

    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        // もし処理が必要ないならスキップ
        if (mask[i]) {
            continue;
        }
        //解答保存領域の確保
        // new (sum_tmp + i) thrust::device_vector<uint32_t>((chunk_len + srclens[i] - 2));

        dim3 grid(((chunk_len + srclens[i] - 2) + block.x - 1) / block.x);
        diffSum<<<grid, block, 0, streams[i]>>>(chunk_d, srcAudios[i], thrust::raw_pointer_cast(sum_tmp[i].data()), chunk_len, srclens[i]);
    }

    cudaDeviceSynchronize();

    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        // もし処理が必要ないならスキップ
        if (mask[i]) {
            result[i][0] = i;
            result[i][1] = UINT32_MAX;
            continue;
        }
        std::cout << "dev pass." << i << std::endl;
        std::cout << sum_tmp[i][0] << std::endl;

        result[i][0] = i;
        result[i][1] = thrust::reduce(
            thrust::device,
            sum_tmp[i].begin(),
            sum_tmp[i].end(),
            UINT32_MAX,
            thrust::minimum<uint32_t>());
        std::cout << "result[" << i << "][1]=" << result[i][1] << std::endl;

        //後片付け
        // sum_tmp[i].~device_vector();
    }
    cudaDeviceSynchronize();

    std::sort((comp_pair *)result, (comp_pair *)result + BASE_AUDIO_N, [](const auto &a, const auto &b) { return a.second < b.second; });

    cudaFree(chunk_d);
}

// DLLのロードアンロードにフックしてる
namespace {
struct LoadFook {
    LoadFook() {
        for (auto i = 0; i < BASE_AUDIO_N; i++) {
            cudaStreamCreate(&streams[i]);
        }

        isInit = true;
    }

    ~LoadFook() {
        if (!isInit) return;
        for (auto i = 0; i < BASE_AUDIO_N; i++) {
            cudaStreamDestroy(streams[i]);
        }
        isInit = false;

        if (srcLoaded) {
            for (auto i = 0; i < BASE_AUDIO_N; i++) {
                cudaFree(srcAudios[i]);
            }
        }
    }
} loadfook;
}  // namespace