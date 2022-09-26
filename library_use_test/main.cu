#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "add_vector.h"

void initialize_list(float *list, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        list[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    float *dest, *lhs, *rhs;
    size_t n = 3;

    size_t bytes_dest = n * sizeof(float);
    size_t bytes_lhs = n * sizeof(float);
    size_t bytes_rhs = n * sizeof(float);

    dest = (float *)malloc(bytes_dest);
    lhs = (float *)malloc(bytes_lhs);
    rhs = (float *)malloc(bytes_rhs);

    srand((unsigned int)time(NULL));
    initialize_list(lhs, n);
    initialize_list(rhs, n);

    add_vector(dest, lhs, rhs, n);

    for (size_t i = 0; i < n; ++i) {
        printf("%.6f + %.6f = %.6f\n", lhs[i], rhs[i], dest[i]);
    }

    free(dest);
    free(lhs);
    free(rhs);
}