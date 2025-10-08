//
//  GCRSKernel.cu
//  NoSynFree
//
//  Created by Liu Chengjian on 17/8/29.
//  Copyright (c) 2017年 csliu. All rights reserved.
//

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "GCRSCommon.h"

__constant__ unsigned int CONST_DEV_BITMATRIX_CODED_INT[MAX_K*MAX_W*4];

extern "C" cudaError_t gcrs_cuda_sm_set_column_coding_bitmatrix(unsigned int *coding_column_bitmatrix,int k,int m,int w, int factor){
    return cudaMemcpyToSymbol(CONST_DEV_BITMATRIX_CODED_INT, coding_column_bitmatrix, k*w*factor*sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
}

__global__ void gcrs_m_1_w_4_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 4;
    int i,j;
    long result = 0;

    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;

    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;

    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w

    unsigned int bitInt = 0x01;
    unsigned int matrixInt;

    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result;
}

__global__ void gcrs_m_1_w_5_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 5;
    int i,j;
    long result = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result;
}

__global__ void gcrs_m_1_w_6_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 6;
    int i,j;
    long result = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result;
}

__global__ void gcrs_m_1_w_7_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 7;
    int i,j;
    long result = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result;
}

__global__ void gcrs_m_1_w_8_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 8;
    int i,j;
    long result = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result = result ^ ( (((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result;
}

__global__ void gcrs_m_2_w_4_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 4;
    int i,j;
    long result[2];
    
    result[0] = 0;
    result[1] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
}

__global__ void gcrs_m_2_w_5_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 5;
    int i,j;
    long result[2];
    
    result[0] = 0;
    result[1] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
}

__global__ void gcrs_m_2_w_6_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 6;
    int i,j;
    long result[2];
    
    result[0] = 0;
    result[1] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
}

__global__ void gcrs_m_2_w_7_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 7;
    int i,j;
    long result[2];
    
    result[0] = 0;
    result[1] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
}

__global__ void gcrs_m_2_w_8_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 8;
    int i,j;
    long result[2];
    
    result[0] = 0;
    result[1] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
}

__global__ void gcrs_m_3_w_4_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 4;

    int i,j;
    long result[3];
    
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);

            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
    out[idx + 2 * size] = result[2];
}

__global__ void gcrs_m_3_w_5_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 5;
    
    int i,j;
    long result[3];
    
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
    out[idx + 2 * size] = result[2];
}

__global__ void gcrs_m_3_w_6_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 6;
    
    int i,j;
    long result[3];
    
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
    out[idx + 2 * size] = result[2];
}

__global__ void gcrs_m_3_w_7_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 7;
    
    int i,j;
    long result[3];
    
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
    out[idx + 2 * size] = result[2];
}

__global__ void gcrs_m_3_w_8_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 8;
    
    int i,j;
    long result[3];
    
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
    out[idx + 2 * size] = result[2];
}

__global__ void gcrs_m_4_w_4_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 4;
    int i,j;
    long result[4];
    
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
    out[idx + 2 * size] = result[2];
    out[idx + 3 * size] = result[3];
}

__global__ void gcrs_m_4_w_5_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 5;
    int i,j;
    long result[4];
    
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
    out[idx + 2 * size] = result[2];
    out[idx + 3 * size] = result[3];
}

__global__ void gcrs_m_4_w_6_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 6;
    int i,j;
    long result[4];
    
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
    out[idx + 2 * size] = result[2];
    out[idx + 3 * size] = result[3];
}

__global__ void gcrs_m_4_w_7_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 7;
    int i,j;
    long result[4];
    
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
    out[idx + 2 * size] = result[2];
    out[idx + 3 * size] = result[3];
}

__global__ void gcrs_m_4_w_8_coding_dotprod(int k, int index, long *in, long *out,  int size){
    extern __shared__ long shared_data[];
    
    int w = 8;
    int i,j;
    long result[4];
    
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
    
    const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFF;
    
    int worksize_perblock = blockDim.x / w * w;
    const unsigned  int idx = worksize_perblock * blockIdx.x + threadIdx.x;
    
    if (threadIdx.x >= worksize_perblock) {
        return;
    }
    
    if (idx >= size) {
        return;
    }
    
    int group_offset = (threadIdx.x / w) * w;
    int group_inner_offset = threadIdx.x % w;
    // row for each thread in the bitmatrix * row size which is k * w
    
    unsigned int bitInt = 0x01;
    unsigned int matrixInt;
    
    for ( i = 0; i < k; i++ ) {
        
        shared_data[threadIdx.x] = *(in + i*size + idx);
        
        __syncthreads();
        
#pragma unroll
        for ( j = 0; j < w; j++ ) {
            matrixInt = CONST_DEV_BITMATRIX_CODED_INT[index];
            result[0] = result[0] ^ ((((matrixInt & (bitInt<< group_inner_offset)) >> group_inner_offset) * fullOneBit) & shared_data[group_offset + j]);
            result[1] = result[1] ^ ((((matrixInt & (bitInt<< (group_inner_offset+w))) >> (group_inner_offset+w)) * fullOneBit) & shared_data[group_offset + j]);
            result[2] = result[2] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 2*w))) >> (group_inner_offset + 2*w)) * fullOneBit) & shared_data[group_offset + j]);
            result[3] = result[3] ^ ((((matrixInt & (bitInt<< (group_inner_offset + 3*w))) >> (group_inner_offset + 3*w)) * fullOneBit) & shared_data[group_offset + j]);
            
            ++index;
        }
        __syncthreads();
        
    }
    
    out[idx] = result[0];
    out[idx + size] = result[1];
    out[idx + 2 * size] = result[2];
    out[idx + 3 * size] = result[3];
}

extern "C" void m_1_w_4_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_1_w_4_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_1_w_5_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_1_w_5_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_1_w_6_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_1_w_6_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_1_w_7_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_1_w_7_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_1_w_8_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_1_w_8_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}


extern "C" void m_2_w_4_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_2_w_4_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_2_w_5_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_2_w_5_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_2_w_6_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_2_w_6_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_2_w_7_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_2_w_7_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}


extern "C" void m_2_w_8_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_2_w_8_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_3_w_4_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_3_w_4_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_3_w_5_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_3_w_5_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_3_w_6_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_3_w_6_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_3_w_7_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_3_w_7_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_3_w_8_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_3_w_8_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_4_w_4_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_4_w_4_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_4_w_5_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_4_w_5_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_4_w_6_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);
    
    gcrs_m_4_w_6_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_4_w_7_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);

    gcrs_m_4_w_7_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}

extern "C" void m_4_w_8_coding(int k, int index,
                                char *dataPtr, char *codeDevPtr,
                                int threadDimX,int blockDimX,
                                int workSizePerGridInLong,
                                cudaStream_t stream){
    dim3 gridDim(blockDimX, 1, 1);
    dim3 blockDim(threadDimX, 1, 1);

    gcrs_m_4_w_8_coding_dotprod<<<gridDim, blockDim, threadDimX*sizeof(long), stream>>>(k, index, (long *)dataPtr, (long *)codeDevPtr, workSizePerGridInLong);
    
}


