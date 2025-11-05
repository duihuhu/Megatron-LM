//
// Test Program: GPU Tensor Zero-Copy Encoding (Configurable)
// Compile: make test_gpu_tensor
// Run: ./test_gpu_tensor [options]
//
// Options:
//   -k <num>          Number of data blocks (default: 4)
//   -m <num>          Number of parity blocks (default: 2)
//   -w <num>          Galois field width (default: 4)
//   -b <size>         Block size in KB (default: 64)
//   -t <num>          Number of tasks for pipelining (default: 1)
//   --threads <num>   Threads per block (default: 128)
//   --blocks <num>    Blocks per grid (default: 0 = auto)
//   --test <num>      Test number to run (1=CPU, 2=GPU zero-copy, 3=manual, 4=XOR, 0=all, default: 0)
//   -h, --help        Show this help message
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "PErasureWorker.h"
#include "GCRSCommon.h"
#include "GCRSCoding.h"

// Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Configuration structure
typedef struct {
    int k;
    int m;
    int w;
    size_t block_size;
    int task_num;
    int threads_per_block;
    int blocks_per_grid;
} TestConfig;

// Default configuration
TestConfig default_config = {
    .k = 4,
    .m = 2,
    .w = 4,
    .block_size = 64 * 1024,  // 64KB
    .task_num = 1,
    .threads_per_block = 128,
    .blocks_per_grid = 0  // auto-calculate
};

void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n\n", prog_name);
    printf("Options:\n");
    printf("  -k <num>          Number of data blocks (default: %d)\n", default_config.k);
    printf("  -m <num>          Number of parity blocks (default: %d)\n", default_config.m);
    printf("  -w <num>          Galois field width (default: %d)\n", default_config.w);
    printf("  -b <size>         Block size in KB (default: %zu)\n", default_config.block_size / 1024);
    printf("  -t <num>          Number of tasks for pipelining (default: %d)\n", default_config.task_num);
    printf("  --threads <num>   Threads per block (default: %d)\n", default_config.threads_per_block);
    printf("  --blocks <num>    Blocks per grid (default: %d = auto)\n", default_config.blocks_per_grid);
    printf("  --test <num>      Test number (1=CPU, 2=GPU zero-copy, 3=manual, 4=XOR, 0=all, default: 0)\n");
    printf("  -h, --help        Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s -k 8 -m 4 -b 128 --threads 256\n", prog_name);
    printf("  %s -k 4 -m 2 -w 8 -t 4 --test 2\n", prog_name);
}

int parse_args(int argc, char **argv, TestConfig *config) {
    *config = default_config;
    int test_num = 0;  // 0 = all tests
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return -1;
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            config->k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            config->m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            config->w = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            config->block_size = atoi(argv[++i]) * 1024;  // Convert KB to bytes
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            config->task_num = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            config->threads_per_block = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            config->blocks_per_grid = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            test_num = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }
    
    return test_num;
}

void print_config(TestConfig *config) {
    printf("Configuration:\n");
    printf("  k (data blocks):        %d\n", config->k);
    printf("  m (parity blocks):      %d\n", config->m);
    printf("  w (Galois field width): %d\n", config->w);
    printf("  block_size:            %zu KB (%zu bytes)\n", config->block_size / 1024, config->block_size);
    printf("  task_num:              %d\n", config->task_num);
    printf("  threads_per_block:     %d\n", config->threads_per_block);
    if (config->blocks_per_grid == 0) {
        printf("  blocks_per_grid:       auto-calculate\n");
    } else {
        printf("  blocks_per_grid:       %d\n", config->blocks_per_grid);
    }
    printf("\n");
}

// Test 1: CPU data -> GPU encoding -> CPU result (traditional mode)
int test_cpu_encoding(TestConfig *config) {
    printf("\n[Test 1] CPU -> GPU -> CPU (Traditional Mode)\n");
    printf("------------------------------------------------\n");
    
    size_t total_input_size = config->block_size * config->k;
    size_t total_output_size = config->block_size * config->m;
    
    // Initialize worker
    struct PErasureWorker *worker = PErasureWorkerInit(config->k, config->m, config->w, config->block_size, config->task_num);
    if (!worker) {
        printf("ERROR: Failed to initialize worker\n");
        return -1;
    }
    
    // Configure CUDA execution parameters
    PErasureWorkerSetThreadsPerBlock(worker, config->threads_per_block);
    if (config->blocks_per_grid > 0) {
        PErasureWorkerSetBlocksPerGrid(worker, config->blocks_per_grid);
    }
    
    printf("✓ Worker initialized: k=%d, m=%d, w=%d, block_size=%zu KB\n", 
           config->k, config->m, config->w, config->block_size / 1024);
    printf("✓ CUDA config: threads_per_block=%d, blocks_per_grid=%d\n",
           PErasureWorkerGetThreadsPerBlock(worker),
           PErasureWorkerGetBlocksPerGrid(worker));
    
    // Allocate and initialize CPU memory
    char *cpu_input = (char *)malloc(total_input_size);
    char *cpu_output = (char *)malloc(total_output_size);
    if (!cpu_input || !cpu_output) {
        printf("ERROR: Failed to allocate CPU memory\n");
        return -1;
    }
    
    // Fill input with test data
    for (size_t i = 0; i < total_input_size; i++) {
        cpu_input[i] = (char)(i % 256);
    }
    printf("✓ CPU input data prepared (%zu bytes)\n", total_input_size);
    
    // Set input data
    PErasureWorkerSetInputData(worker, cpu_input, total_input_size);
    
    // Encode
    printf("Running encoding...\n");
    fullDuplexRunEncode(worker);
    
    // Get results
    PErasureWorkerGetOutputData(worker, cpu_output, total_output_size);
    
    // Verify results (simple check: output should not be all zeros)
    int all_zero = 1;
    for (size_t i = 0; i < total_output_size && all_zero; i++) {
        if (cpu_output[i] != 0) all_zero = 0;
    }
    
    if (all_zero) {
        printf("WARNING: Output appears to be all zeros\n");
    } else {
        printf("✓ Encoding completed successfully\n");
    }
    
    // Calculate throughput
    PErasureWorkerCalculateRecord(worker);
    double encode_time = PErasureWorkerGetEncodeTimeConsume(worker);
    double throughput = (double)total_input_size / encode_time / 1024.0 / 1024.0 * 1000.0;  // MB/s
    printf("  Encoding time: %.3f ms\n", encode_time);
    printf("  Throughput: %.2f MB/s\n", throughput);
    
    // Clean up
    free(cpu_input);
    free(cpu_output);
    PErasureWorkerDealloc(worker);
    
    return 0;
}

// Test 2: GPU tensor zero-copy encoding
int test_gpu_zero_copy(TestConfig *config) {
    printf("\n[Test 2] GPU Tensor -> GPU Encoding -> GPU Tensor (Zero-Copy Mode)\n");
    printf("-------------------------------------------------------------------\n");
    
    size_t total_input_size = config->block_size * config->k;
    size_t total_output_size = config->block_size * config->m;
    
    // Initialize worker
    struct PErasureWorker *worker = PErasureWorkerInit(config->k, config->m, config->w, config->block_size, config->task_num);
    if (!worker) {
        printf("ERROR: Failed to initialize worker\n");
        return -1;
    }
    
    // Configure CUDA execution parameters
    PErasureWorkerSetThreadsPerBlock(worker, config->threads_per_block);
    if (config->blocks_per_grid > 0) {
        PErasureWorkerSetBlocksPerGrid(worker, config->blocks_per_grid);
    }
    
    printf("✓ Worker initialized: k=%d, m=%d, w=%d, block_size=%zu KB\n", 
           config->k, config->m, config->w, config->block_size / 1024);
    printf("✓ CUDA config: threads_per_block=%d, blocks_per_grid=%d\n",
           PErasureWorkerGetThreadsPerBlock(worker),
           PErasureWorkerGetBlocksPerGrid(worker));
    
    // Allocate GPU memory
    char *gpu_input = NULL;
    char *gpu_output = NULL;
    // Align buffer size to match worker's internal alignment
    size_t aligned_block_size = ((config->block_size + sizeof(long) * config->w - 1) / (sizeof(long) * config->w)) * (sizeof(long) * config->w);
    size_t total_input_size_aligned = aligned_block_size * config->k;
    size_t total_output_size_aligned = aligned_block_size * config->m;
    
    CUDA_CHECK(cudaMalloc((void **)&gpu_input, total_input_size_aligned));
    CUDA_CHECK(cudaMalloc((void **)&gpu_output, total_output_size_aligned));
    printf("✓ GPU memory allocated (aligned: input=%zu KB, output=%zu KB)\n", 
           total_input_size_aligned / 1024, total_output_size_aligned / 1024);
    
    // Initialize GPU input data
    char *cpu_temp = (char *)malloc(total_input_size);
    for (size_t i = 0; i < total_input_size; i++) {
        cpu_temp[i] = (char)(i % 256);
    }
    CUDA_CHECK(cudaMemcpy(gpu_input, cpu_temp, total_input_size, cudaMemcpyHostToDevice));
    free(cpu_temp);
    printf("✓ GPU input data prepared (%zu bytes)\n", total_input_size);
    
    // Set up zero-copy mode - use aligned sizes
    PErasureWorkerEncodeGPUZeroCopy(worker, gpu_input, gpu_output, total_input_size_aligned);
    printf("✓ Zero-copy mode enabled (no H2D/D2H transfers)\n");
    
    // Run encoding
    printf("Running encoding (zero-copy mode)...\n");
    fullDuplexRunEncode(worker);
    
    // Verify results by copying back to CPU (just for verification)
    char *cpu_result = (char *)malloc(total_output_size_aligned);
    CUDA_CHECK(cudaMemcpy(cpu_result, gpu_output, total_output_size_aligned, cudaMemcpyDeviceToHost));
    
    int all_zero = 1;
    for (size_t i = 0; i < total_output_size && all_zero; i++) {
        if (cpu_result[i] != 0) all_zero = 0;
    }
    
    if (all_zero) {
        printf("WARNING: Output appears to be all zeros\n");
    } else {
        printf("✓ Zero-copy encoding completed successfully\n");
    }
    
    // Calculate throughput
    PErasureWorkerCalculateRecord(worker);
    double encode_time = PErasureWorkerGetEncodeTimeConsume(worker);
    double throughput = (double)total_input_size / encode_time / 1024.0 / 1024.0 * 1000.0;  // MB/s
    printf("  Encoding time: %.3f ms\n", encode_time);
    printf("  Throughput: %.2f MB/s\n", throughput);
    printf("  Note: Zero-copy mode eliminates PCIe transfer overhead!\n");
    
    // Clean up
    CUDA_CHECK(cudaFree(gpu_input));
    CUDA_CHECK(cudaFree(gpu_output));
    free(cpu_result);
    PErasureWorkerDealloc(worker);
    
    return 0;
}

// Test 3: Manual zero-copy setup
int test_manual_zero_copy(TestConfig *config) {
    printf("\n[Test 3] Manual Zero-Copy Setup\n");
    printf("--------------------------------\n");
    
    size_t total_input_size = config->block_size * config->k;
    size_t total_output_size = config->block_size * config->m;
    
    struct PErasureWorker *worker = PErasureWorkerInit(config->k, config->m, config->w, config->block_size, config->task_num);
    if (!worker) {
        printf("ERROR: Failed to initialize worker\n");
        return -1;
    }
    
    // Configure CUDA execution parameters
    PErasureWorkerSetThreadsPerBlock(worker, config->threads_per_block);
    if (config->blocks_per_grid > 0) {
        PErasureWorkerSetBlocksPerGrid(worker, config->blocks_per_grid);
    }
    
    printf("✓ Worker initialized: k=%d, m=%d, w=%d, block_size=%zu KB\n", 
           config->k, config->m, config->w, config->block_size / 1024);
    printf("✓ CUDA config: threads_per_block=%d, blocks_per_grid=%d\n",
           PErasureWorkerGetThreadsPerBlock(worker),
           PErasureWorkerGetBlocksPerGrid(worker));
    
    // Allocate GPU buffers with proper alignment
    char *gpu_input = NULL;
    char *gpu_output = NULL;
    // Align buffer size to match worker's internal alignment
    size_t aligned_block_size = ((config->block_size + sizeof(long) * config->w - 1) / (sizeof(long) * config->w)) * (sizeof(long) * config->w);
    size_t total_input_size_aligned = aligned_block_size * config->k;
    size_t total_output_size_aligned = aligned_block_size * config->m;
    
    CUDA_CHECK(cudaMalloc((void **)&gpu_input, total_input_size_aligned));
    CUDA_CHECK(cudaMalloc((void **)&gpu_output, total_output_size_aligned));
    
    // Initialize GPU input
    char *cpu_temp = (char *)malloc(total_input_size);
    for (size_t i = 0; i < total_input_size; i++) {
        cpu_temp[i] = (char)(i % 256);
    }
    CUDA_CHECK(cudaMemcpy(gpu_input, cpu_temp, total_input_size, cudaMemcpyHostToDevice));
    free(cpu_temp);
    
    // Manual setup - use aligned sizes
    PErasureWorkerSetInputDevicePtr(worker, gpu_input, total_input_size_aligned);
    PErasureWorkerSetOutputDevicePtr(worker, gpu_output, total_output_size_aligned);
    PErasureWorkerSetSkipD2HTransfer(worker, 1);
    printf("✓ Manual zero-copy setup completed\n");
    
    // Run encoding
    printf("Running encoding...\n");
    fullDuplexRunEncode(worker);
    
    // Get output device pointer
    char *output_ptr = PErasureWorkerGetOutputDevicePtr(worker);
    if (output_ptr == gpu_output) {
        printf("✓ Output device pointer verified\n");
    }
    
    // Calculate throughput
    PErasureWorkerCalculateRecord(worker);
    double encode_time = PErasureWorkerGetEncodeTimeConsume(worker);
    double throughput = (double)total_input_size / encode_time / 1024.0 / 1024.0 * 1000.0;  // MB/s
    printf("✓ Manual zero-copy encoding completed\n");
    printf("  Encoding time: %.3f ms\n", encode_time);
    printf("  Throughput: %.2f MB/s\n", throughput);
    
    // Clean up
    CUDA_CHECK(cudaFree(gpu_input));
    CUDA_CHECK(cudaFree(gpu_output));
    PErasureWorkerDealloc(worker);
    
    return 0;
}

// Test 4: XOR operation test (simple XOR of k input blocks)
int test_xor_operation(TestConfig *config) {
    printf("\n[Test 4] XOR Operation Test\n");
    printf("---------------------------\n");
    
    size_t total_input_size = config->block_size * config->k;
    size_t total_output_size = config->block_size;  // XOR produces one output block
    
    printf("✓ Configuration: k=%d, block_size=%zu KB\n", config->k, config->block_size / 1024);
    
    // Allocate GPU memory
    char *gpu_input = NULL;
    char *gpu_output = NULL;
    CUDA_CHECK(cudaMalloc((void **)&gpu_input, total_input_size));
    CUDA_CHECK(cudaMalloc((void **)&gpu_output, total_output_size));
    printf("✓ GPU memory allocated (input=%zu KB, output=%zu KB)\n", 
           total_input_size / 1024, total_output_size / 1024);
    
    // Initialize GPU input data
    char *cpu_input = (char *)malloc(total_input_size);
    for (size_t i = 0; i < total_input_size; i++) {
        cpu_input[i] = (char)(i % 256);
    }
    CUDA_CHECK(cudaMemcpy(gpu_input, cpu_input, total_input_size, cudaMemcpyHostToDevice));
    free(cpu_input);
    printf("✓ GPU input data prepared (%zu bytes)\n", total_input_size);
    
    // Calculate thread and block configuration
    int threadNum = config->threads_per_block;
    size_t workSizePerBlock = threadNum * sizeof(long);
    int blockNum = config->block_size / workSizePerBlock;
    if (config->block_size % workSizePerBlock != 0) {
        blockNum++;
    }
    
    // Override with custom blocks_per_grid if set
    if (config->blocks_per_grid > 0) {
        blockNum = config->blocks_per_grid;
    }
    
    int workSizeInLong = config->block_size / sizeof(long);
    
    printf("✓ CUDA config: threads_per_block=%d, blocks_per_grid=%d\n", threadNum, blockNum);
    
    // Warm-up run
    printf("Running warm-up...\n");
    gcrs_xor_coding(config->k, 0, gpu_input, gpu_output, threadNum, blockNum, workSizeInLong, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run XOR operation multiple times for accurate timing
    int num_iterations = 10;
    float total_time = 0.0;
    
    // Create CUDA events once (reuse for all iterations)
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    
    printf("Running XOR operation (%d iterations)...\n", num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        float time_elapsed;
        CUDA_CHECK(cudaEventRecord(startEvent, 0));
        gcrs_xor_coding(config->k, 0, gpu_input, gpu_output, threadNum, blockNum, workSizeInLong, 0);
        CUDA_CHECK(cudaEventRecord(stopEvent, 0));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
        CUDA_CHECK(cudaEventElapsedTime(&time_elapsed, startEvent, stopEvent));
        total_time += time_elapsed;
    }
    
    // Destroy events
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    
    float avg_time = total_time / num_iterations;
    double throughput = (double)total_input_size / avg_time / 1024.0 / 1024.0 * 1000.0;  // MB/s
    
    printf("✓ XOR operation completed successfully\n");
    printf("  Average time (%d iterations): %.3f ms\n", num_iterations, avg_time);
    printf("  Throughput: %.2f MB/s\n", throughput);
    printf("  Note: XOR is simpler than EC encoding, should be faster!\n");
    
    // Verify result by copying back to CPU
    char *cpu_output = (char *)malloc(total_output_size);
    CUDA_CHECK(cudaMemcpy(cpu_output, gpu_output, total_output_size, cudaMemcpyDeviceToHost));
    
    // Verify: XOR result should not be all zeros (unless all inputs are identical)
    int all_zero = 1;
    for (size_t i = 0; i < total_output_size && all_zero; i++) {
        if (cpu_output[i] != 0) all_zero = 0;
    }
    
    if (all_zero) {
        printf("  WARNING: XOR result is all zeros (inputs might be identical)\n");
    } else {
        printf("  ✓ XOR result verified (non-zero output)\n");
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(gpu_input));
    CUDA_CHECK(cudaFree(gpu_output));
    free(cpu_output);
    
    return 0;
}

int main(int argc, char **argv) {
    printf("==================================================\n");
    printf("G-CRSPCIE GPU Tensor Zero-Copy Encoding Tests\n");
    printf("==================================================\n");
    
    // Parse command line arguments
    TestConfig config;
    int test_num = parse_args(argc, argv, &config);
    if (test_num == -1) {
        return 1;  // Help was shown or error occurred
    }
    
    // Print configuration
    print_config(&config);
    
    // Check CUDA availability
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("ERROR: No CUDA devices found\n");
        return 1;
    }
    printf("Found %d CUDA device(s)\n", device_count);
    
    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("Using device: %s\n", prop.name);
    printf("\n");
    
    int success = 0;
    int total_tests = 0;
    
    // Run tests based on test_num
    if (test_num == 0 || test_num == 1) {
        total_tests++;
        if (test_cpu_encoding(&config) == 0) {
            success++;
        }
    }
    
    if (test_num == 0 || test_num == 2) {
        total_tests++;
        if (test_gpu_zero_copy(&config) == 0) {
            success++;
        }
    }
    
    if (test_num == 0 || test_num == 3) {
        total_tests++;
        if (test_manual_zero_copy(&config) == 0) {
            success++;
        }
    }
    
    if (test_num == 0 || test_num == 4) {
        total_tests++;
        if (test_xor_operation(&config) == 0) {
            success++;
        }
    }
    
    printf("\n==================================================\n");
    printf("Test Summary: %d/%d tests passed\n", success, total_tests);
    printf("==================================================\n");
    
    return (success == total_tests) ? 0 : 1;
}
