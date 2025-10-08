//
//  PCIEMeasure.c
//  GCRSPCIE
//
//  Created by Liu Chengjian on 17/10/9.
//  Copyright (c) 2017年 csliu. All rights reserved.
//

#include <stdio.h>

#include "GCRSCommon.h"
#include "GCRSCoding.h"
#include "GCRSMatrix.h"

#include "PErasureUtilities.h"
#include "PErasureWorker.h"

#define ARG_SIZE 4
void Usage(){
    printf("Input:./Program m workSizePerDataParityBlockInMB numberOfTasks\n");
    exit(0);
}

int main(int argc, const char * argv[]) {
    // insert code here...
    int k, m, w;
    int workSizePerDataParityBlock, numOfTasks;
    
    if (argc != ARG_SIZE) {
        Usage();
    }
    
    m = atoi(argv[1]);
    workSizePerDataParityBlock = atoi(argv[2]) *1024 *1024;
    numOfTasks = atoi(argv[3]);
    
    printf("k \t EncodeTime \t EnThroughput \t DecodeTime \t DeThroughput \t sizePerBlockInMB\n");
    
    //MAX_K
    for (k = m; k <= MAX_K; ++k) {
        w = 4;
        w = gcrs_check_k_m_w(k, m, w);
        //        printf("k:%d, m:%d w:%d\n",k,m,w);
        int *bitmatrix = gcrs_create_bitmatrix(k, m, w);
        //        printMatrix(bitmatrix, k*w, m*w);
//        unsigned int *column_bitmatrix = gcrs_create_column_coding_bitmatrix(k, m, w, bitmatrix);
        int *erasures = talloc(int, k+m+1);
        int erasureNum = randomErasureM(erasures, m, k+m);
        int *erased = gcrs_erasures_to_erased(k, m, erasures);
        int *matIdx = talloc(int, erasureNum);
        int *dm_ids = talloc(int, k);
        int *decoding_bitmatrix = gcrs_create_decoding_bitmatrix(k, m, w, bitmatrix,
                                                                     matIdx, erasures, dm_ids);
        //        printMatrix(decoding_bitmatrix, k*w, erasureNum*w);
        
//        unsigned int *decoding_column_bitmatrix = gcrs_create_column_coding_bitmatrix(k, m, w, decoding_bitmatrix);
        
        
        struct PErasureWorker *worker = PErasureWorkerInit(k, m, w, workSizePerDataParityBlock, numOfTasks);
        GCRSMCodingInit(&worker->mCoding, k, m, w);
        GCRSMCodingSetBitmatrix(&worker->mCoding, bitmatrix);
        //        GCRS_cuda_sm_set_column_encoded_bitmatrix(column_bitmatrix, k, m, w, 2);
        //        cudaError_t error = GCRS_cuda_set_column_bitmatrix(column_bitmatrix, k, w, 0);
        //        if (error != cudaSuccess) {
        //            printf("error:%s when call cudaMemcpySymbol\n",cudaGetErrorString(error));
        //        }
        fullDuplexRunEncode(worker);
        
        int datablockMissing = 0;
        int kIdx;
        for (kIdx = 0; kIdx < k; ++kIdx) {
            if (erased[kIdx] == 1) {
                datablockMissing = 1;
            }
        }
        
        if (datablockMissing == 0) {
            for (kIdx = 0; kIdx < k; ++kIdx) {
                dm_ids[kIdx] = kIdx;
            }
        }
        
        //        producerConsumerRun(worker);
        
        setupDecoding(worker, dm_ids);
        GCRSMCodingDealloc(&worker->mCoding);
        GCRSMCodingInit(&worker->mCoding, k, erasureNum, w);
        GCRSMCodingSetBitmatrix(&worker->mCoding, decoding_bitmatrix);
        //        GCRS_cuda_sm_set_column_encoded_bitmatrix(decoding_column_bitmatrix, k, erasureNum, w, 2);
        fullDuplexRunDecode(worker);
        
        checkEncodeDecodeResult(worker, matIdx, erasureNum);
        PErasureWorkerCalculateRecord(worker);
        //        GCRSWorkerPrintRecord(worker);
        double encodeTimeConsume = PErasureWorkerGetEncodeTimeConsume(worker);
        double encodeThroughput = (double)((k)* worker->bufSize)/encodeTimeConsume*1000.0/1024.0/1024.0/1024.0;
        double decodeTimeConsume = PErasureWorkerGetDecodeTimeConsume(worker);
        double decodeThroughput = (double)((k)* worker->bufSize)/decodeTimeConsume*1000.0/1024.0/1024.0/1024.0;
        
        printf("%d \t %f \t %f \t %f \t %f \t %f\n",
               k, encodeTimeConsume, encodeThroughput, decodeTimeConsume, decodeThroughput, ((double)((k)* worker->bufSize)/1024.0/1024.0));
        GCRSMCodingDealloc(&worker->mCoding);
        PErasureWorkerDealloc(worker);
        
        free(bitmatrix);
        free(erasures);
        free(erased);
        free(matIdx);
        free(dm_ids);
        free(decoding_bitmatrix);
//        free(column_bitmatrix);
//        free(decoding_column_bitmatrix);
        
        free(worker);
        
        //        return 0;
    }
    
    return 0;
}