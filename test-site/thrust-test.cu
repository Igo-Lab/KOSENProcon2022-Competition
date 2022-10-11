#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <new>
#include <random>

__host__ static __inline__ uint32_t rand_01() {
    static std::mt19937_64 mt64(0);

    // [min_val, max_val] の一様分布整数 (int) の分布生成器
    std::uniform_int_distribution<uint64_t> get_rand_uni_int(0, 5000);

    // 乱数を生成
    int t = get_rand_uni_int(mt64);
    std::cout << t << std::endl;
    return t;
}

__global__ void test(unsigned int *arr, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%u\n", arr[idx]);
}

int main() {
    {
        int lens[] = {10, 5};
        thrust::device_vector<unsigned int> sum_tmpd[88];

        for (auto i = 0; i < 88; i++) {
            printf("%d ", lens[i % 2]);
            // new (sum_tmpd + i) thrust::device_vector<unsigned int>(lens[i % 2]);
            sum_tmpd[i].resize(lens[i % 2]);
            thrust::sequence(thrust::device, sum_tmpd[i].begin(), sum_tmpd[i].end());
            unsigned int sum = thrust::reduce(thrust::device, sum_tmpd[i].begin(), sum_tmpd[i].end(), UINT32_MAX, thrust::minimum<unsigned int>());
            thrust::host_vector<uint32_t> htmp(sum_tmpd[i]);
            printf("%u %u %u\n", htmp[0], htmp[lens[i % 2] - 1], sum);
        }

        dim3 grid(1);
        test<<<5, 1>>>(thrust::raw_pointer_cast(sum_tmpd[1].data()), 5);
    }

    // {
    //     thrust::device_vector<uint32_t> sumtmp;
    //     sumtmp.resize(10);
    //     thrust::sequence(thrust::device, sumtmp.begin(), sumtmp.end());
    //     printf("%u %u\n", sumtmp[0], sumtmp[9]);
    // }
    return 0;
}