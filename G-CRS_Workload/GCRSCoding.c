//
//  GCRSCoding.c
//  NoSynFree
//
//  Created by Liu Chengjian on 17/8/29.
//  Copyright (c) 2017年 csliu. All rights reserved.
//

#include "GCRSCoding.h"
#include "GCRSCommon.h"

void (*gcrs_coders[])(char *dataPtr, char *codeDevPtr,
                      int threadDimX,int blockDimX,
                      int k,
                      int workSizePerGridInLong)
= { gcrs_nosynfree_m_1_w_4_coding, gcrs_nosynfree_m_1_w_5_coding, gcrs_nosynfree_m_1_w_6_coding,gcrs_nosynfree_m_1_w_7_coding, gcrs_nosynfree_m_1_w_8_coding,
    gcrs_nosynfree_m_2_w_4_coding, gcrs_nosynfree_m_2_w_5_coding, gcrs_nosynfree_m_2_w_6_coding,gcrs_nosynfree_m_2_w_7_coding, gcrs_nosynfree_m_2_w_8_coding,
    gcrs_nosynfree_m_3_w_4_coding, gcrs_nosynfree_m_3_w_5_coding, gcrs_nosynfree_m_3_w_6_coding,gcrs_nosynfree_m_3_w_7_coding, gcrs_nosynfree_m_3_w_8_coding,
    gcrs_nosynfree_m_4_w_4_coding, gcrs_nosynfree_m_4_w_5_coding, gcrs_nosynfree_m_4_w_6_coding,gcrs_nosynfree_m_4_w_7_coding, gcrs_nosynfree_m_4_w_8_coding
};


void (* coding_ptr)(char *dataPtr, char *codeDevPtr,
                    int threadDimX,int blockDimX,
                    int k,
                    int workSizePerGridInLong);

void assign_coding_ptr(int m, int w){
    if (m < MIN_M || m > MAX_M || w< MIN_W || w > MAX_W) {
        return;
    }
    
    coding_ptr = gcrs_coders[(m - 1) * (MAX_W - MIN_W + 1) + (w- MIN_W)];
}

void gcrs_raw_coding_measure(int k, int m, int w,
                             int threadDimX, int blockDimX,
                             int workSizeTotalInLong,
                             char *devDataPtr, char *devCodePtr, float *timeElasped){
    
    assign_coding_ptr(m, w);
    
    double timeElapsed, throughputInMB;
    cudaEvent_t startEvent, stopEvent;
    
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    cudaEventRecord(startEvent,0);
    
    coding_ptr(devDataPtr, devCodePtr,
            threadDimX, blockDimX,
            k,
            workSizeTotalInLong);
    
    cudaEventRecord(stopEvent,0);
    cudaThreadSynchronize();
    cudaEventSynchronize(startEvent); //optional
    cudaEventSynchronize(stopEvent); //wait for the event to be executed!
    
    cudaEventElapsedTime(timeElasped, startEvent, stopEvent);
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}
