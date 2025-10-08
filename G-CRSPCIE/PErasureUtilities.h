//
//  PErasureUtilities.h
//  PErasurePCIE
//
//  Created by Liu Chengjian on 16/10/19.
//  Copyright (c) 2016年 csliu. All rights reserved.
//

#ifndef __PErasurePCIE__PErasureUtilities__
#define __PErasurePCIE__PErasureUtilities__

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void alloc_cuda_host_memory(void **ptr, size_t size);
void alloc_cuda_device_memory(void **ptr, size_t size);

void free_cuda_host_memory(void *ptr);
void free_cuda_device_memory(void *ptr);

void transfer_cuda_memory_host_to_device_async(void* dst, const void* src, size_t count, cudaStream_t stream);
void transfer_cuda_memory_device_to_host_async(void* dst, const void* src, size_t count, cudaStream_t stream);

size_t align_value(size_t valueToAlign, size_t alignMask);

void generateRandomValue(char *data, size_t size);
int randomErasureM(int *erasures, int m, int length);

void create_cuda_stream(cudaStream_t *pStream);
void sync_cuda_stream(cudaStream_t stream);
void destroy_cuda_stream(cudaStream_t * pStream);

double get_elapsed_time_in_ms(struct timeval startTime, struct timeval endTime);

#endif /* defined(__PErasurePCIE__PErasureUtilities__) */
