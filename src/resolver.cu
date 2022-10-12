
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

constexpr size_t BASE_AUDIO_N = 88;
constexpr size_t BLOCK_N = 256;

typedef struct {
    uint32_t index;
    uint32_t value;
} pair;

int16_t *srcAudios[BASE_AUDIO_N];  // gpuのメモリポインタ．マルチスレッドで呼び出すときに問題になりそう．
int32_t srclens[BASE_AUDIO_N];
cudaStream_t streams[BASE_AUDIO_N];
bool isInit = false;
bool srcLoaded = false;

__global__ void diffSum(const int16_t *__restrict__ chunk, const int16_t *__restrict__ src, uint32_t *sums, const int32_t chunkLen, const int32_t sourceLen) {
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

__global__ void printest(int16_t *arr, uint32_t len) {
    for (auto i = 0; i < len; i++) {
        printf("%d\n", arr[i]);
    }
}

__global__ void argtest(const int16_t *__restrict__ chunk, const int16_t *__restrict__ src, uint32_t *sums, const int32_t chunkLen, const int32_t sourceLen) {
    printf("chlen: %d, srclen: %d\n", chunkLen, sourceLen);

    for (auto i = 0; i < 10; i++) {
        for (auto j = 0; j < 10; j++) {
            printf("%d ", chunk[i * 10 + j]);
        }
        printf("\n");
    }
    printf("\n");
    for (auto i = 0; i < 10; i++) {
        for (auto j = 0; j < 10; j++) {
            printf("%d ", src[i * 10 + j]);
        }
        printf("\n");
    }
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
        CUDA_SAFE_CALL(cudaMalloc((void **)&(srcAudios[i]), sizeof(int16_t) * lens[i]));
        // 読みデータのコピー
        CUDA_SAFE_CALL(cudaMemcpy(srcAudios[i], srcs[i], sizeof(int16_t) * lens[i], cudaMemcpyHostToDevice));

        srclens[i] = lens[i];
    }

    srcLoaded = true;
}

// 元読みデータはindex-0スタート．つまり0～87
void resolver(const int16_t *chunk, const int32_t chunk_len, const bool *mask, uint32_t **result_raw) {
    int16_t *chunk_d;
    thrust::device_vector<uint32_t> sum_tmp[BASE_AUDIO_N];
    dim3 block(BLOCK_N);

    pair *result = (pair *)result_raw;

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

        // if(i == 0){
        //     printf("chlen: %u srclen: %u", chunk_len, srclens[0]);
        //     argtest<<<1, 1>>>(chunk_d, srcAudios[i], thrust::raw_pointer_cast(sum_tmp[0].data()), chunk_len, srclens[0]);
        // }

        dim3 grid(((chunk_len + srclens[i] - 2) + block.x - 1) / block.x);
        printf("srcAudio ID: %d, Block: %d, Grid: %d\n", i + 1, block.x, grid.x);
        printf("sums_d size: %d, %d\n", sum_tmp[i].size(), grid.x * block.x);
        diffSum<<<grid, block, 0, streams[i]>>>(chunk_d, srcAudios[i], thrust::raw_pointer_cast(sum_tmp[i].data()), chunk_len, srclens[i]);
    }

    cudaDeviceSynchronize();

    for (auto i = 0; i < BASE_AUDIO_N; i++) {
        // もし処理が必要ないならスキップ
        if (mask[i]) {
            result[i].index = i;
            result[i].value = UINT32_MAX;
            continue;
        }
        std::cout << "dev pass." << i << std::endl;

        // uint32_t itmp = i;
        // uint32_t redtmp = thrust::reduce(
        //     thrust::device,
        //     sum_tmp[i].begin(),
        //     sum_tmp[i].end(),
        //     UINT32_MAX,
        //     thrust::minimum<uint32_t>());

        result[i].index = i;
        result[i].value = thrust::reduce(
            thrust::device,
            sum_tmp[i].begin(),
            sum_tmp[i].end(),
            UINT32_MAX,
            thrust::minimum<uint32_t>());
    }

    cudaDeviceSynchronize();

    std::sort(result, result + BASE_AUDIO_N, [](const auto &a, const auto &b) { return a.value < b.value; });

    for (auto j = 0; j < BASE_AUDIO_N; j++) {
        printf("[%u, %u],", result[j].index + 1, result[j].value);
    }

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