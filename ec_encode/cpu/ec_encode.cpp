#include <stdio.h>
#include <chrono>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <isa-l.h>

#define KB (1024UL)
#define MB (1024UL * KB)
#define GB (1024UL * MB)

// Utility to measure seconds, returns as double
static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void run_encode(int k, int m, size_t len) {
    // Allocate data and parity pointers
    uint8_t **data = (uint8_t **)malloc(sizeof(uint8_t *) * k);
    uint8_t **parity = (uint8_t **)malloc(sizeof(uint8_t *) * m);
    for (int i = 0; i < k; ++i) {
        data[i] = (uint8_t *)malloc(len);
        memset(data[i], 0, len);
    }
    for (int i = 0; i < m; ++i) {
        parity[i] = (uint8_t *)malloc(len);
        memset(parity[i], 0, len);
    }
    // Generate encode matrix (Vandermonde)
    uint8_t *encode_matrix = (uint8_t *)malloc((k+m) * k);
    gf_gen_rs_matrix(encode_matrix, k + m, k);

    // Set up gftbls
    uint8_t *gftbls = (uint8_t *)malloc(k * m * 32);
    ec_init_tables(k, m, &encode_matrix[k * k], gftbls);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    size_t max_len = (size_t)INT32_MAX;
    if (len <= max_len) {
        ec_encode_data((int)len, k, m, gftbls, data, parity);
    } else {
        size_t offset = 0;
        while (offset < len) {
            size_t chunk_size = (len - offset > max_len) ? max_len : (len - offset);
            uint8_t **data_ptrs = (uint8_t **)malloc(sizeof(uint8_t *) * k);
            uint8_t **parity_ptrs = (uint8_t **)malloc(sizeof(uint8_t *) * m);
            for (int i = 0; i < k; ++i)
                data_ptrs[i] = data[i] + offset;
            for (int i = 0; i < m; ++i)
                parity_ptrs[i] = parity[i] + offset;
            ec_encode_data((int)chunk_size, k, m, gftbls, data_ptrs, parity_ptrs);
            free(data_ptrs);
            free(parity_ptrs);
            offset += chunk_size;
        }
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    printf("Encode (k=%d, m=%d, total=%.2f GB): ", k, m, (double)(k * len) / GB);
    printf("Time %.4fs, Throughput: %.2f GB/s\n", std::chrono::duration<double>(t2-t1).count(), (k*len)/(GB*std::chrono::duration<double>(t2-t1).count()));

    for (int i = 0; i < k; ++i) free(data[i]);
    for (int i = 0; i < m; ++i) free(parity[i]);
    free(data);
    free(parity);
    free(encode_matrix);
    free(gftbls);
}

int main() {
    size_t sizes[] = {512*MB, 1*GB, 2*GB, 4*GB, 8*GB, 16*GB};
    int num_sizes = sizeof(sizes)/sizeof(sizes[0]);

    struct { int k, m; } cases[] = { {2,2}, {6,2}, {14,2} };
    int num_cases = sizeof(cases)/sizeof(cases[0]);

    for (int si = 0; si < num_sizes; ++si) {
        size_t len = sizes[si];
        for (int ci = 0; ci < num_cases; ++ci) {
            int k = cases[ci].k;
            int m = cases[ci].m;
            // Check if the memory can be allocated for data and parity blocks before running encode
            size_t data_block_size = len / k;
            uint8_t **data_test = (uint8_t **)malloc(sizeof(uint8_t *) * k);
            uint8_t **parity_test = (uint8_t **)malloc(sizeof(uint8_t *) * m);
            bool alloc_failed = false;
            if (!data_test || !parity_test) {
                alloc_failed = true;
            } else {
                for (int di = 0; di < k; ++di) {
                    data_test[di] = (uint8_t *)malloc(data_block_size);
                    if (!data_test[di]) {
                        alloc_failed = true;
                        break;
                    }
                }
                for (int pi = 0; pi < m && !alloc_failed; ++pi) {
                    parity_test[pi] = (uint8_t *)malloc(data_block_size);
                    if (!parity_test[pi]) {
                        alloc_failed = true;
                        break;
                    }
                }
            }
            if (data_test) {
                for (int di = 0; di < k; ++di) {
                    if (data_test[di]) free(data_test[di]);
                }
                free(data_test);
            }
            if (parity_test) {
                for (int pi = 0; pi < m; ++pi) {
                    if (parity_test[pi]) free(parity_test[pi]);
                }
                free(parity_test);
            }
            if (alloc_failed) {
                printf("Memory allocation failed for (k=%d, m=%d, size=%.2f GB), skip this test.\n", k, m, (double)len / GB);
                continue;
            }
            run_encode(k, m, data_block_size);
        }
    }

    return 0;
}