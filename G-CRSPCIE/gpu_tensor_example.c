//
// Example: GPU Tensor Zero-Copy Encoding
// This example demonstrates how to use G-CRSPCIE for encoding GPU tensors
// without CPU-GPU memory transfers (zero-copy mode)
//

#include <stdio.h>
#include <cuda_runtime.h>
#include "PErasureWorker.h"
#include "GCRSCommon.h"

// Example 1: CPU data -> GPU encoding -> CPU result (traditional mode)
void example_cpu_to_gpu_encoding() {
    printf("=== Example 1: CPU -> GPU -> CPU (Traditional Mode) ===\n");
    
    int k = 4, m = 2, w = 4;
    size_t block_size = 1024 * 1024;  // 1MB per block
    size_t total_size = block_size * k;
    int task_num = 1;
    
    // Initialize worker
    struct PErasureWorker *worker = PErasureWorkerInit(k, m, w, block_size, task_num);
    if (!worker) {
        printf("Failed to initialize worker\n");
        return;
    }
    
    // Allocate CPU memory for input data
    char *cpu_input = (char *)malloc(total_size);
    // ... fill cpu_input with your data ...
    
    // Set input data (this will be copied to GPU during encoding)
    PErasureWorkerSetInputData(worker, cpu_input, total_size);
    
    // Encode (H2D transfer happens, then encoding, then D2H transfer)
    fullDuplexRunEncode(worker);
    
    // Get results back to CPU
    char *cpu_output = (char *)malloc(block_size * m);
    PErasureWorkerGetOutputData(worker, cpu_output, block_size * m);
    
    // Clean up
    free(cpu_input);
    free(cpu_output);
    PErasureWorkerDealloc(worker);
    printf("Traditional encoding completed\n\n");
}

// Example 2: GPU tensor -> GPU encoding -> GPU tensor (zero-copy mode)
void example_gpu_tensor_zero_copy() {
    printf("=== Example 2: GPU Tensor -> GPU Encoding -> GPU Tensor (Zero-Copy Mode) ===\n");
    
    int k = 4, m = 2, w = 4;
    size_t block_size = 1024 * 1024;  // 1MB per block
    size_t total_size = block_size * k;
    int task_num = 1;
    
    // Initialize worker
    struct PErasureWorker *worker = PErasureWorkerInit(k, m, w, block_size, task_num);
    if (!worker) {
        printf("Failed to initialize worker\n");
        return;
    }
    
    // Allocate GPU memory for input data (e.g., from PyTorch tensor or CUDA)
    char *gpu_input = NULL;
    cudaMalloc((void **)&gpu_input, total_size);
    // ... fill gpu_input with your GPU data (or use existing GPU tensor pointer) ...
    
    // Allocate GPU memory for output (e.g., from PyTorch tensor or CUDA)
    char *gpu_output = NULL;
    cudaMalloc((void **)&gpu_output, block_size * m);
    
    // Method 1: Use the convenient API
    PErasureWorkerEncodeGPUZeroCopy(worker, gpu_input, gpu_output, total_size);
    
    // Now run encoding (no H2D or D2H transfers!)
    fullDuplexRunEncode(worker);
    
    // Results are directly in gpu_output, no need to copy back
    // You can use gpu_output directly in your GPU application
    
    // Clean up
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    PErasureWorkerDealloc(worker);
    printf("Zero-copy GPU encoding completed\n\n");
}

// Example 3: Manual setup for zero-copy mode
void example_manual_zero_copy_setup() {
    printf("=== Example 3: Manual Zero-Copy Setup ===\n");
    
    int k = 4, m = 2, w = 4;
    size_t block_size = 1024 * 1024;
    size_t total_size = block_size * k;
    int task_num = 1;
    
    struct PErasureWorker *worker = PErasureWorkerInit(k, m, w, block_size, task_num);
    if (!worker) {
        printf("Failed to initialize worker\n");
        return;
    }
    
    // Allocate GPU buffers
    char *gpu_input = NULL;
    char *gpu_output = NULL;
    cudaMalloc((void **)&gpu_input, total_size);
    cudaMalloc((void **)&gpu_output, block_size * m);
    
    // Method 2: Manual setup (more control)
    // Set input device pointer (skips H2D transfer)
    PErasureWorkerSetInputDevicePtr(worker, gpu_input, total_size);
    
    // Set output device pointer (skips D2H transfer)
    PErasureWorkerSetOutputDevicePtr(worker, gpu_output, block_size * m);
    
    // Enable skip D2H transfer flag
    PErasureWorkerSetSkipD2HTransfer(worker, 1);
    
    // Run encoding
    fullDuplexRunEncode(worker);
    
    // Get output device pointer if needed
    char *output_ptr = PErasureWorkerGetOutputDevicePtr(worker);
    // output_ptr == gpu_output (already set above)
    
    // Clean up
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    PErasureWorkerDealloc(worker);
    printf("Manual zero-copy encoding completed\n\n");
}

// Example 4: Using with PyTorch tensors (pseudo-code)
// In Python/C++ interop, you would do:
/*
void example_pytorch_integration() {
    // In Python:
    // import torch
    // data_tensor = torch.randint(0, 256, (k, block_size), dtype=torch.uint8, device='cuda')
    // output_tensor = torch.zeros((m, block_size), dtype=torch.uint8, device='cuda')
    
    // Get device pointers from PyTorch tensors
    // char *input_ptr = (char *)data_tensor.data_ptr();
    // char *output_ptr = (char *)output_tensor.data_ptr();
    
    // Use zero-copy encoding
    // PErasureWorkerEncodeGPUZeroCopy(worker, input_ptr, output_ptr, total_size);
    // fullDuplexRunEncode(worker);
    
    // Results are directly in output_tensor, no CPU round-trip needed!
}
*/

int main() {
    printf("G-CRSPCIE GPU Tensor Zero-Copy Encoding Examples\n");
    printf("==================================================\n\n");
    
    // Run examples
    example_cpu_to_gpu_encoding();
    example_gpu_tensor_zero_copy();
    example_manual_zero_copy_setup();
    
    return 0;
}

