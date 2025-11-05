//
//  GCRSCoding.c
//  NoSynFree
//
//  Created by Liu Chengjian on 17/8/29.
//  Copyright (c) 2017年 csliu. All rights reserved.
//

#include "GCRSCoding.h"
#include "GCRSCommon.h"
#include "GCRSMatrix.h"


void (*coding_func_array[])(int k, int index,
                            char *dataPtr, char *codeDevPtr,
                            int threadDimX,int blockDimX,
                            int workSizePerGridInLong,
                            cudaStream_t stream) = {
    m_1_w_4_coding,m_1_w_5_coding,m_1_w_6_coding,m_1_w_7_coding,m_1_w_8_coding,
    m_2_w_4_coding,m_2_w_5_coding,m_2_w_6_coding,m_2_w_7_coding,m_2_w_8_coding,
    m_3_w_4_coding,m_3_w_5_coding,m_3_w_6_coding,m_3_w_7_coding,m_3_w_8_coding,
    m_4_w_4_coding,m_4_w_5_coding,m_4_w_6_coding,m_4_w_7_coding,m_4_w_8_coding
};

//void (* coding_ptr)(char *dataPtr, char *codeDevPtr,
//                    int threadDimX,int blockDimX,
//                    int k,
//                    int workSizePerGridInLong);
//
//void assign_coding_ptr(int m, int w){
//    if (m < MIN_M || m > MAX_M || w< MIN_W || w > MAX_W) {
//        return;
//    }
//    
//    coding_ptr = gcrs_coders[(m - 1) * (MAX_W - MIN_W + 1) + (w- MIN_W)];
//}

void GCRSMCodingInit(struct GCRSMCoding * mCoding, int k, int m, int w){
    mCoding->k = k;
    mCoding->m = m;
    mCoding->w = w;
    
    if (m < MAX_M) {
        mCoding->taskSize = 1;
        mCoding->mValue = talloc(int, mCoding->taskSize);
        mCoding->index = talloc(int, mCoding->taskSize);
        
        mCoding->index[0] = 0;
        mCoding->mValue[0] = m;
        mCoding->coding_function_ptrs = talloc(coding_func, 1);
    }else{
        int taskNum = m / MAX_M, mRemain = m;
        int idx;
        if (m % MAX_M != 0) {
            ++taskNum;
        }
        
        mCoding->taskSize = taskNum;
        mCoding->index = talloc(int, mCoding->taskSize);
        mCoding->mValue = talloc(int, mCoding->taskSize);
        mCoding->coding_function_ptrs = talloc(coding_func, mCoding->taskSize);
        for (idx = 0; idx < taskNum; ++idx) {
            
            if (mRemain < MAX_M) {
                mCoding->mValue[idx] = mRemain;
            }else{
                mCoding->mValue[idx] = MAX_M;
                mRemain = mRemain - MAX_M;
            }
            
            if (idx == 0) {
                mCoding->index[idx] = 0;
            }else{
                mCoding->index[idx] = mCoding->index[idx-1] + mCoding->k*mCoding->w;
            }
        }
    }
    
    GCRSMCodingInitFuncPtrs(mCoding);
}

void GCRSMCodingInitFuncPtrs(struct GCRSMCoding * mCoding){
    int idx;
    
    for (idx = 0; idx < mCoding->taskSize; ++idx) {
        //        printf("idx:%d, function idx:%d\n",idx,((mCoding->mValue[idx] - 1) * (MAX_W - MIN_W + 1)+ mCoding->w - MIN_W));
        mCoding->coding_function_ptrs[idx] = coding_func_array[(mCoding->mValue[idx] - 1) * (MAX_W - MIN_W + 1)+ mCoding->w - MIN_W];
    }
}

void GCRSMCodingSetBitmatrix(struct GCRSMCoding * mCoding,
                             int *bitmatrix){
    int idx;
    int mValue = 0;
    unsigned int *all_columns_bitmatrix = talloc(unsigned int, (mCoding->k * mCoding->w* mCoding->taskSize));
    
    //    printMatrix(bitmatrix, mCoding->k*mCoding->w, mCoding->k*mCoding->m );
    //    printf("\n************************\n");
    
    for (idx = 0; idx < mCoding->taskSize; ++idx) {
        unsigned int *column_bitmatrix = gcrs_create_column_coding_bitmatrix(mCoding->k, mCoding->mValue[idx], mCoding->w, bitmatrix+mCoding->k*mCoding->w*mValue*mCoding->w);
        //        perasure_print_column_encoded_bitmatrix(column_bitmatrix, mCoding->k, 1, mCoding->w);
        memcpy((all_columns_bitmatrix + idx * mCoding->k * mCoding->w), column_bitmatrix, mCoding->k*mCoding->w * sizeof(unsigned int));
        
        free(column_bitmatrix);
        mValue = mValue + mCoding->mValue[idx];
        
    }
    
    cudaError_t error =  gcrs_cuda_sm_set_column_coding_bitmatrix(all_columns_bitmatrix, mCoding->k, mCoding->m , mCoding->w, mCoding->taskSize);
    
    if (error != cudaSuccess) {
        printf("error:%s when call perasure_cuda_sm_set_column_encoded_bitmatrix k:%d m:%d w:%d taskSize:%d\n",cudaGetErrorString(error), mCoding->k, mCoding->m , mCoding->w, mCoding->taskSize);
    }
    
    
    //    PErasureMCodingVerifyBitmatrix(mCoding, bitmatrix, all_columns_bitmatrix);
    
    free(all_columns_bitmatrix);
}

void GCRSMCodingCall(struct GCRSMCoding * mCoding,
                     char *dataPtr, char *codePtr,
                     int threadDimX,int blockDimX,
                     int workSizePerGridInLong,
                     cudaStream_t stream){
    int idx;
    int size = workSizePerGridInLong * sizeof(long);
    int mValue = 0;
    for (idx = 0; idx < mCoding->taskSize; ++idx) {
        //        printf("size:%d, mValue:%d, index:%d\n",size, mValue,mCoding->index[idx] );
        mCoding->coding_function_ptrs[idx](mCoding->k, mCoding->index[idx],
                                           dataPtr, (codePtr + mValue * size),
                                           threadDimX,blockDimX,
                                           workSizePerGridInLong,
                                           stream);
        mValue = mValue + mCoding->mValue[idx];
    }
}

float GCRSMCodingMeasure(struct GCRSMCoding * mCoding,
                         char *dataPtr, char *codePtr,
                         int threadDimX,int blockDimX,
                         int workSizePerGridInLong){
    cudaEvent_t startEvent, stopEvent;
    
    int idx;
    int size = workSizePerGridInLong * sizeof(long);
    int mValue = 0;
    
    float elapsedTime;
    
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    cudaEventRecord(startEvent, 0);
    for (idx = 0; idx < mCoding->taskSize; ++idx) {
        //        printf("size:%d, mValue:%d, index:%d\n",size, mValue,mCoding->index[idx] );
        mCoding->coding_function_ptrs[idx](mCoding->k, mCoding->index[idx],
                                           dataPtr, (codePtr + mValue * size),
                                           threadDimX,blockDimX,
                                           workSizePerGridInLong,
                                           0);
        mValue = mValue + mCoding->mValue[idx];
    }
    
    cudaEventRecord(stopEvent,0);
    cudaThreadSynchronize();
    cudaEventSynchronize(startEvent); //optional
    cudaEventSynchronize(stopEvent); //wait for the event to be executed!
    
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    
    return elapsedTime;
}

void GCRSMCodingDealloc(struct GCRSMCoding * mCoding){
    free(mCoding->mValue);
    free(mCoding->index);
    free(mCoding->coding_function_ptrs);
}

void GCRSMCodingVerifyBitmatrix(struct GCRSMCoding * mCoding,
                                int *bitmatrix, unsigned int *column_bitmatrix){
    int *column_coverted_bitmatrx;
    int taskIdx, kIdx, wIdx, wInnerIdx, mAdded,mIdx;
    
    column_coverted_bitmatrx = talloc(int, mCoding->k * mCoding->w * mCoding->m * mCoding->w);
    
    unsigned int bitOne = 0x01;
    mAdded = 0;
    for (taskIdx = 0 ; taskIdx < mCoding->taskSize; ++taskIdx) {
        for (kIdx = 0; kIdx < mCoding->k; ++kIdx) {
            for (wIdx = 0; wIdx < mCoding->w; ++wIdx) {
                unsigned int value = column_bitmatrix[taskIdx*mCoding->k*mCoding->w + kIdx * mCoding->w + wIdx];
                //                printf("\nvalue:%d\n",value);
                for (mIdx = 0; mIdx < mCoding->mValue[taskIdx]; ++mIdx) {
                    for (wInnerIdx = 0; wInnerIdx < mCoding->w; ++wInnerIdx) {
                        //                        printf("idx:%d ", (mAdded*mCoding->k*mCoding->m*mCoding->w + (mIdx*mCoding->w + wInnerIdx)*mCoding->k*mCoding->w + kIdx*mCoding->w + wIdx));
                        if (((value & (bitOne << ( mIdx * mCoding->w + wInnerIdx))) >> ( mIdx * mCoding->w + wInnerIdx)) == bitOne) {
                            column_coverted_bitmatrx[mAdded*mCoding->w*mCoding->k*mCoding->w + (mIdx*mCoding->w + wInnerIdx)*mCoding->k*mCoding->w + kIdx*mCoding->w + wIdx] = 1;
                        }else{
                            column_coverted_bitmatrx[mAdded*mCoding->w*mCoding->k*mCoding->w + (mIdx*mCoding->w + wInnerIdx)*mCoding->k*mCoding->w + kIdx*mCoding->w + wIdx] = 0;
                        }
                    }
                }
            }
        }
        mAdded = mAdded + mCoding->mValue[taskIdx];
    }
    
    printf("\n*******************\n");
    printMatrix(bitmatrix, mCoding->k * mCoding->w, mCoding->w * mCoding->m);
    printf("\n*******************\n");
    printMatrix(column_coverted_bitmatrx, mCoding->k * mCoding->w, mCoding->w * mCoding->m);
    printf("\n*******************\n");
    
}

// XOR measurement function
void gcrs_xor_measure(int k,
                      int threadDimX, int blockDimX,
                      int workSizeTotalInLong,
                      char *devDataPtr, char *devCodePtr, float *timeElapsed) {
    cudaEvent_t startEvent, stopEvent;
    
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    cudaEventRecord(startEvent, 0);
    gcrs_xor_coding(k, 0, devDataPtr, devCodePtr, threadDimX, blockDimX, workSizeTotalInLong, 0);
    cudaEventRecord(stopEvent, 0);
    
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(timeElapsed, startEvent, stopEvent);
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

