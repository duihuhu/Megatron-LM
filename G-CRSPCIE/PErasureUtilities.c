//
//  PErasureUtilities.cpp
//  PErasurePCIE
//
//  Created by Liu Chengjian on 16/10/19.
//  Copyright (c) 2016年 csliu. All rights reserved.
//

#include "PErasureUtilities.h"

#include <time.h>

void alloc_cuda_host_memory(void **ptr, size_t size){
    cudaError_t error = cudaMallocHost(ptr, size);
    if(error != cudaSuccess){
        printf("error:%s when call cudaMallocHost\n",cudaGetErrorString(error));
    }
}

void alloc_cuda_device_memory(void **ptr, size_t size){
    cudaError_t error = cudaMalloc(ptr, size);
    if(error != cudaSuccess){
        printf("error:%s when call cudaMalloc\n",cudaGetErrorString(error));
    }
}

void free_cuda_host_memory(void *ptr){
    cudaError_t error =  cudaFreeHost(ptr);
    if(error != cudaSuccess){
        printf("error:%s when call cudaFreeHost\n",cudaGetErrorString(error));
    }
}

void free_cuda_device_memory(void *ptr){
    cudaError_t error = cudaFree(ptr);
    if(error != cudaSuccess){
        printf("error:%s when call cudaFree\n",cudaGetErrorString(error));
    }
}

void transfer_cuda_memory_host_to_device_async(void* dst, const void* src, size_t count, cudaStream_t stream){
    cudaError_t error =  cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
    if(error != cudaSuccess){
        printf("error:%s when call cudaMemcpyAsync cudaMemcpyHostToDevice\n",cudaGetErrorString(error));
    }
}

void transfer_cuda_memory_device_to_host_async(void* dst, const void* src, size_t count, cudaStream_t stream){
    cudaError_t error =  cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
    if(error != cudaSuccess){
        printf("error:%s when call cudaMemcpyAsync cudaMemcpyDeviceToHost\n",cudaGetErrorString(error));
    }
}

size_t align_value(size_t valueToAlign, size_t alignMask){
    if (valueToAlign % alignMask != 0) {
        valueToAlign = valueToAlign + (alignMask - valueToAlign % alignMask);
    }
    
    return valueToAlign;
}

char randomChar(){
    return rand() % 256;
}

void generateRandomValue(char *data, size_t size){
    int idx;
    
    srand(time(NULL));
    
    for (idx = 0; idx < size; ++idx) {
        *(data+idx) = randomChar();
    }
}

int randomErasureM(int *erasures, int m, int length){
    int erasureNum = 0;
    int mIdx;
    
    srand(time(NULL));
    
    do{
        for (mIdx = 0; mIdx < m; ++mIdx) {
            
            if (erasureNum == m) {
                break;
            }
            int erasureID = rand() % length;
            if (erasureNum == 0) {
                // This assure at least one erasured
                erasures[erasureNum] = erasureID;
                ++erasureNum;
            }else{
                int erasureIdx;
                for (erasureIdx = 0; erasureIdx < erasureNum; ++erasureIdx) {
                    if (erasures[erasureIdx] == erasureID) {
                        break;
                    }
                }
                
                if (erasureIdx == erasureNum) {
                    erasures[erasureNum] = erasureID;
                    ++erasureNum;
                }
            }
        }
    }while (erasureNum < m);
    
    erasures[erasureNum] = -1;
    
    return erasureNum;
}

void create_cuda_stream(cudaStream_t * 	pStream){
    cudaError_t error = cudaStreamCreate(pStream);
    if(error != cudaSuccess){
        printf("error:%s when call cudaStreamCreate\n",cudaGetErrorString(error));
    }
}

void sync_cuda_stream(cudaStream_t stream){
    cudaStreamSynchronize(stream);
}

void destroy_cuda_stream(cudaStream_t * pStream){
    cudaError_t error = cudaStreamDestroy(*pStream);
    
    if(error != cudaSuccess){
        printf("error:%s when call cudaStreamDestroy\n",cudaGetErrorString(error));
    }
}

double get_elapsed_time_in_ms(struct timeval startTime, struct timeval endTime){
    double timeElapsed = ((double)(endTime.tv_sec - startTime.tv_sec)) * 1000.0 + ((double)(endTime.tv_usec - startTime.tv_usec))/ 1000.0;
    return timeElapsed;
}

