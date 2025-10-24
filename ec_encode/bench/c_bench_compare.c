/* c_bench_compare.c
 * Small C benchmark to compare isa-l's xor_gen vs xor_gen_base and
 * ec_encode_data vs ec_encode_data_base.
 *
 * Build (from repo root):
 *   gcc -O3 -march=native -Iisa-l/include -Lisa-l/lib -o c_bench_compare \
 *       ec_encode/bench/c_bench_compare.c -lisal -pthread
 *
 * You may need to adjust -L and -I to point to the built libisal and headers.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "erasure_code.h"
#include "raid.h"

/* globals for threaded workers */
static unsigned char **g_big_v = NULL; /* for xor: per-vect big buffers */
static unsigned char **g_big_src = NULL; /* for ec: per-src big buffers */
static unsigned char **g_big_dest = NULL; /* for ec: per-dest big buffers */
static unsigned char *g_gftbls = NULL;
static int g_vects = 0;
static int g_len = 0;
static int g_iters = 0;
static int g_warmup = 0;
static int g_use_base = 0;
static int g_k = 0;
static int g_rows = 0;

static void *aligned_alloc32(size_t size) {
    void *ptr = NULL;
    int rc = posix_memalign(&ptr, 32, size);
    if (rc != 0) return NULL;
    return ptr;
}

static void fill_random(unsigned char *p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = rand() & 0xFF;
}

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *xor_worker_thread(void *arg) {
    int id = *(int *)arg;
    int curcpu = sched_getcpu();
    printf("xor worker %d started on cpu %d\n", id, curcpu);
    unsigned char **local = malloc(sizeof(unsigned char*) * g_vects);
    for (int i = 0; i < g_vects; ++i) local[i] = g_big_v[i] + (size_t)id * g_len;
    for (int w = 0; w < g_warmup; ++w) {
        if (g_use_base) xor_gen_base(g_vects, g_len, (void **)local);
        else xor_gen(g_vects, g_len, (void **)local);
    }
    double t0 = now_sec();
    for (int it = 0; it < g_iters; ++it) {
        if (g_use_base) xor_gen_base(g_vects, g_len, (void **)local);
        else xor_gen(g_vects, g_len, (void **)local);
    }
    double t1 = now_sec();
    double *ret = malloc(sizeof(double));
    *ret = t1 - t0;
    free(local);
    return ret;
}

static void *ec_worker_thread(void *arg) {
    int id = *(int *)arg;
    int curcpu = sched_getcpu();
    printf("ec worker %d started on cpu %d\n", id, curcpu);
    unsigned char **src_local = malloc(sizeof(unsigned char*) * g_k);
    unsigned char **dest_local = malloc(sizeof(unsigned char*) * g_rows);
    for (int i = 0; i < g_k; ++i) src_local[i] = g_big_src[i] + (size_t)id * g_len;
    for (int i = 0; i < g_rows; ++i) dest_local[i] = g_big_dest[i] + (size_t)id * g_len;
    for (int w = 0; w < g_warmup; ++w) {
        if (g_use_base) ec_encode_data_base(g_len, g_k, g_rows, g_gftbls, src_local, dest_local);
        else ec_encode_data(g_len, g_k, g_rows, g_gftbls, src_local, dest_local);
    }
    double t0 = now_sec();
    for (int it = 0; it < g_iters; ++it) {
        if (g_use_base) ec_encode_data_base(g_len, g_k, g_rows, g_gftbls, src_local, dest_local);
        else ec_encode_data(g_len, g_k, g_rows, g_gftbls, src_local, dest_local);
    }
    double t1 = now_sec();
    double *ret = malloc(sizeof(double));
    *ret = t1 - t0;
    free(src_local); free(dest_local);
    return ret;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s xor|ec [--base] -k <k> -d <dests> -n <len> -i <iters> -w <warmup> --workers <n>\n", argv[0]);
        return 1;
    }

    const char *mode = argv[1];
    int use_base = 0;
    int k = 2;
    int dests = 1;
    int len = 1024*1024;
    int iters = 100;
    int warmup = 5;
    int workers = 1;
    int do_pin = 0;
    int cpu_start = 0;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--base") == 0) use_base = 1;
        else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) k = atoi(argv[++i]);
        else if (strcmp(argv[i], "-d") == 0 && i+1 < argc) dests = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) len = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-w") == 0 && i+1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--workers") == 0 && i+1 < argc) workers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--pin") == 0) do_pin = 1;
        else if (strcmp(argv[i], "--cpu-start") == 0 && i+1 < argc) cpu_start = atoi(argv[++i]);
    }

    srand(12345);

    if (strcmp(mode, "xor") == 0) {
        int vects = k + 1; // k data + 1 dest
        if (workers <= 1) {
            unsigned char **ptrs = malloc(sizeof(unsigned char*) * vects);
            unsigned char **bufs = malloc(sizeof(unsigned char*) * vects);
            for (int i = 0; i < vects; ++i) {
                bufs[i] = aligned_alloc32(len);
                if (!bufs[i]) { perror("posix_memalign"); return 1; }
                fill_random(bufs[i], len);
                ptrs[i] = bufs[i];
            }

            // Warmup
            for (int w = 0; w < warmup; ++w) {
                if (use_base) xor_gen_base(vects, len, (void **)ptrs);
                else xor_gen(vects, len, (void **)ptrs);
            }

            double t0 = now_sec();
            for (int it = 0; it < iters; ++it) {
                if (use_base) xor_gen_base(vects, len, (void **)ptrs);
                else xor_gen(vects, len, (void **)ptrs);
            }
            double t1 = now_sec();
            long long total = (long long)len * (long long)k * (long long)iters;
            double mb = total / (1024.0*1024.0);
            double wall = t1 - t0;
            printf("xor mode (%s) k=%d len=%d iters=%d -> bytes=%lld MB=%.2f wall=%.6fs MB/s=%.2f\n",
                   use_base?"base":"opt", k, len, iters, total, mb, wall, mb / wall);

            for (int i = 0; i < vects; ++i) free(bufs[i]);
            free(bufs); free(ptrs);
        } else {
            /* multi-worker mode: allocate per-vector big buffers of size len * workers */
            g_vects = vects; g_len = len; g_iters = iters; g_warmup = warmup; g_use_base = use_base;
            g_big_v = malloc(sizeof(unsigned char*) * vects);
            for (int i = 0; i < vects; ++i) {
                size_t sz = (size_t)len * (size_t)workers;
                g_big_v[i] = aligned_alloc32(sz);
                if (!g_big_v[i]) { perror("posix_memalign big_v"); return 1; }
                /* fill each worker segment */
                for (int w = 0; w < workers; ++w) fill_random(g_big_v[i] + (size_t)w * len, len);
            }
            pthread_t *ths = malloc(sizeof(pthread_t) * workers);
            int *ids = malloc(sizeof(int) * workers);
            for (int id = 0; id < workers; ++id) {
                ids[id] = id;
                if (do_pin) {
                    pthread_attr_t attr;
                    cpu_set_t cpuset;
                    pthread_attr_init(&attr);
                    CPU_ZERO(&cpuset);
                    CPU_SET(cpu_start + id, &cpuset);
                    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
                    if (pthread_create(&ths[id], &attr, xor_worker_thread, &ids[id]) != 0) {
                        perror("pthread_create"); return 1;
                    }
                    pthread_attr_destroy(&attr);
                } else {
                    if (pthread_create(&ths[id], NULL, xor_worker_thread, &ids[id]) != 0) {
                        perror("pthread_create"); return 1;
                    }
                }
            }
            double max_time = 0.0;
            for (int id = 0; id < workers; ++id) {
                void *r;
                pthread_join(ths[id], &r);
                double t = *(double*)r; free(r);
                if (t > max_time) max_time = t;
            }
            long long total = (long long)len * (long long)k * (long long)iters * (long long)workers;
            double mb = total / (1024.0*1024.0);
            double wall = max_time;
            printf("xor mode (%s) k=%d len=%d iters=%d workers=%d -> bytes=%lld MB=%.2f wall=%.6fs MB/s=%.2f\n",
                   use_base?"base":"opt", k, len, iters, workers, total, mb, wall, mb / wall);

            for (int i = 0; i < vects; ++i) free(g_big_v[i]);
            free(g_big_v); free(ths); free(ids);
        }

    } else if (strcmp(mode, "ec") == 0) {
        int rows = dests; // number of dests
        if (workers <= 1) {
            unsigned char **src = malloc(sizeof(unsigned char*) * k);
            unsigned char **dest = malloc(sizeof(unsigned char*) * rows);
            for (int i = 0; i < k; ++i) {
                src[i] = aligned_alloc32(len);
                if (!src[i]) { perror("posix_memalign"); return 1; }
                fill_random(src[i], len);
            }
            for (int i = 0; i < rows; ++i) {
                dest[i] = aligned_alloc32(len);
                if (!dest[i]) { perror("posix_memalign"); return 1; }
                memset(dest[i], 0, len);
            }

            // prepare encode matrix a (m x k) where m = k + rows
            int m = k + rows;
            unsigned char *a = malloc((size_t)k * m);
            for (int i = 0; i < k * m; ++i) a[i] = 1;

            // allocate gftbls buffer: 32 * k * rows
            size_t gtbls_size = 32 * (size_t)k * (size_t)rows;
            unsigned char *g_tbls = aligned_alloc32(gtbls_size);
            if (!g_tbls) { perror("posix_memalign g_tbls"); return 1; }

            // init tables
            ec_init_tables(k, rows, a, g_tbls);

            // Warmup
            for (int w = 0; w < warmup; ++w) {
                if (use_base) ec_encode_data_base(len, k, rows, g_tbls, src, dest);
                else ec_encode_data(len, k, rows, g_tbls, src, dest);
            }

            double t0 = now_sec();
            for (int it = 0; it < iters; ++it) {
                if (use_base) ec_encode_data_base(len, k, rows, g_tbls, src, dest);
                else ec_encode_data(len, k, rows, g_tbls, src, dest);
            }
            double t1 = now_sec();

            long long total = (long long)len * (long long)k * (long long)iters;
            double mb = total / (1024.0*1024.0);
            double wall = t1 - t0;
            printf("ec mode (%s) k=%d rows=%d len=%d iters=%d -> bytes=%lld MB=%.2f wall=%.6fs MB/s=%.2f\n",
                   use_base?"base":"opt", k, rows, len, iters, total, mb, wall, mb / wall);

            for (int i = 0; i < k; ++i) free(src[i]);
            for (int i = 0; i < rows; ++i) free(dest[i]);
            free(src); free(dest); free(a); free(g_tbls);
        } else {
            /* multi-worker ec: allocate per-src and per-dest big buffers */
            g_k = k; g_rows = rows; g_len = len; g_iters = iters; g_warmup = warmup; g_use_base = use_base;
            size_t seg = (size_t)len * (size_t)workers;
            g_big_src = malloc(sizeof(unsigned char*) * k);
            g_big_dest = malloc(sizeof(unsigned char*) * rows);
            for (int i = 0; i < k; ++i) {
                g_big_src[i] = aligned_alloc32(seg);
                if (!g_big_src[i]) { perror("posix_memalign big_src"); return 1; }
                for (int w = 0; w < workers; ++w) fill_random(g_big_src[i] + (size_t)w * len, len);
            }
            for (int i = 0; i < rows; ++i) {
                g_big_dest[i] = aligned_alloc32(seg);
                if (!g_big_dest[i]) { perror("posix_memalign big_dest"); return 1; }
                for (int w = 0; w < workers; ++w) memset(g_big_dest[i] + (size_t)w * len, 0, len);
            }

            // prepare encode matrix a and g_tbls (shared read-only)
            int m = k + rows;
            unsigned char *a = malloc((size_t)k * m);
            for (int i = 0; i < k * m; ++i) a[i] = 1;
            size_t gtbls_size = 32 * (size_t)k * (size_t)rows;
            unsigned char *g_tbls = aligned_alloc32(gtbls_size);
            if (!g_tbls) { perror("posix_memalign g_tbls"); return 1; }
            ec_init_tables(k, rows, a, g_tbls);
            g_gftbls = g_tbls;

            pthread_t *ths = malloc(sizeof(pthread_t) * workers);
            int *ids = malloc(sizeof(int) * workers);
            for (int id = 0; id < workers; ++id) {
                ids[id] = id;
                if (do_pin) {
                    pthread_attr_t attr;
                    cpu_set_t cpuset;
                    pthread_attr_init(&attr);
                    CPU_ZERO(&cpuset);
                    CPU_SET(cpu_start + id, &cpuset);
                    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
                    if (pthread_create(&ths[id], &attr, ec_worker_thread, &ids[id]) != 0) { perror("pthread_create"); return 1; }
                    pthread_attr_destroy(&attr);
                } else {
                    if (pthread_create(&ths[id], NULL, ec_worker_thread, &ids[id]) != 0) { perror("pthread_create"); return 1; }
                }
            }
            double max_time = 0.0;
            for (int id = 0; id < workers; ++id) {
                void *r; pthread_join(ths[id], &r);
                double t = *(double*)r; free(r);
                if (t > max_time) max_time = t;
            }
            long long total = (long long)len * (long long)k * (long long)iters * (long long)workers;
            double mb = total / (1024.0*1024.0);
            double wall = max_time;
            printf("ec mode (%s) k=%d rows=%d len=%d iters=%d workers=%d -> bytes=%lld MB=%.2f wall=%.6fs MB/s=%.2f\n",
                   use_base?"base":"opt", k, rows, len, iters, workers, total, mb, wall, mb / wall);

            for (int i = 0; i < k; ++i) free(g_big_src[i]);
            for (int i = 0; i < rows; ++i) free(g_big_dest[i]);
            free(g_big_src); free(g_big_dest); free(a); free(g_tbls); free(ths); free(ids);
        }

    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 1;
    }

    return 0;
}
