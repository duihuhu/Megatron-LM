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

void gcrs_nosynfree_m_1_w_4_coding(char *dataPtr, char *codeDevPtr,
                          int threadDimX,int blockDimX,
                          int k,
                          int workSizePerGridInLong);
void gcrs_nosynfree_m_1_w_5_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);

void gcrs_nosynfree_m_1_w_6_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);
void gcrs_nosynfree_m_1_w_7_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);
void gcrs_nosynfree_m_1_w_8_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);

void gcrs_nosynfree_m_2_w_4_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);
void gcrs_nosynfree_m_2_w_5_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);

void gcrs_nosynfree_m_2_w_6_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);
void gcrs_nosynfree_m_2_w_7_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);
void gcrs_nosynfree_m_2_w_8_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);

void gcrs_nosynfree_m_3_w_4_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);
void gcrs_nosynfree_m_3_w_5_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);

void gcrs_nosynfree_m_3_w_6_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);
void gcrs_nosynfree_m_3_w_7_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);
void gcrs_nosynfree_m_3_w_8_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);


void gcrs_nosynfree_m_4_w_4_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);
void gcrs_nosynfree_m_4_w_5_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);

void gcrs_nosynfree_m_4_w_6_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);
void gcrs_nosynfree_m_4_w_7_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);
void gcrs_nosynfree_m_4_w_8_coding(char *dataPtr, char *codeDevPtr,
                                   int threadDimX,int blockDimX,
                                   int k,
                                   int workSizePerGridInLong);


void gcrs_raw_coding_measure(int k, int m, int w,
                             int threadDimX, int blockDimX,
                             int workSizeTotalInLong,
                             char *devDataPtr, char *devCodePtr, float *timeElasped);

#endif /* defined(__NoSynFree__GCRSCoding__) */
