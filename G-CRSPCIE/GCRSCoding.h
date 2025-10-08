//
//  GCRSCoding.h
//  NoSynFree
//
//  Created by Liu Chengjian on 17/8/29.
//  Copyright (c) 2017年 csliu. All rights reserved.
//

#ifndef __NoSynFree__GCRSCoding__
#define __NoSynFree__GCRSCoding__

#include <stdio.h>
#include <cuda_runtime.h>

cudaError_t gcrs_cuda_sm_set_column_coding_bitmatrix(unsigned int *coding_column_bitmatrix,int k,int m,int w, int factor);

 void m_1_w_4_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_1_w_5_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_1_w_6_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_1_w_7_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_1_w_8_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_2_w_4_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_2_w_5_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_2_w_6_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_2_w_7_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_2_w_8_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_3_w_4_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_3_w_5_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_3_w_6_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_3_w_7_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_3_w_8_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_4_w_4_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_4_w_5_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_4_w_6_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_4_w_7_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);
 void m_4_w_8_coding(int k, int index,
                               char *dataPtr, char *codeDevPtr,
                               int threadDimX,int blockDimX,
                               int workSizePerGridInLong,
                               cudaStream_t stream);

typedef void (*coding_func)(int k, int index,
    char *dataPtr, char *codeDevPtr,
    int threadDimX,int blockDimX,
    int workSizePerGridInLong,
    cudaStream_t stream);

struct GCRSMCoding{
    int k;
    int m;
    int w;
    int *mValue;
    int *index;
    int taskSize;
    
    coding_func *coding_function_ptrs;
};

void GCRSMCodingInit(struct GCRSMCoding * mCoding, int k, int m, int w);
void GCRSMCodingInitFuncPtrs(struct GCRSMCoding * mCoding);
void GCRSMCodingSetBitmatrix(struct GCRSMCoding * mCoding,
                                 int *bitmatrix);

void GCRSMCodingCall(struct GCRSMCoding * mCoding,
                         char *dataPtr, char *codePtr,
                         int threadDimX,int blockDimX,
                         int workSizePerGridInLong,
                         cudaStream_t stream);

float GCRSMCodingMeasure(struct GCRSMCoding * mCoding,
                             char *dataPtr, char *codePtr,
                             int threadDimX,int blockDimX,
                             int workSizePerGridInLong);

void GCRSMCodingDealloc(struct GCRSMCoding * mCoding);

void GCRSMCodingVerifyBitmatrix(struct GCRSMCoding * mCoding,
                                    int *bitmatrix, unsigned int *column_bitmatrix);

void gcrs_raw_coding_measure(int k, int m, int w,
                             int threadDimX, int blockDimX,
                             int workSizeTotalInLong,
                             char *devDataPtr, char *devCodePtr, float *timeElasped);

#endif /* defined(__NoSynFree__GCRSCoding__) */
