// Online C++ compiler to run C++ program online
#include <algorithm>
#include <iostream>

typedef struct {
    uint32_t index;
    uint32_t value;
} pair;

void printsorted(uint32_t **arr) {
    std::sort((pair *)arr, ((pair *)arr) + 3, [](const auto &a, const auto &b) { return a.value < b.value; });

    pair *tmpp = (pair *)arr;

    for (auto i = 0; i < 4; i++) {
        printf("%u %u\n", tmpp[i].index, tmpp[i].value);
    }
}

int main() {
    // Write C++ code here
    uint32_t a[][2] = {{1, 1000}, {2, 500}, {3, 400}, {4, 600}};
    uint32_t **b = (uint32_t **)a;
    printsorted(b);
    std::cout << "Hello world!";

    return 0;
}