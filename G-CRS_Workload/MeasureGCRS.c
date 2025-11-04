//
//  MeasureGCRS.c
//  NoBitmatrixOpt
//
//  Created by Liu Chengjian on 17/8/28.
//  Copyright (c) 2017年 csliu. All rights reserved.
//

#include "MeasureGCRS.h"

#include "GCRSCoding.h"

#include "GCRSCommon.h"
#include "GCRSMatrix.h"

#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

float calMean(float *valueArray, int size){
    float avgValue = 0.0;
    int idx = 0;
    
    for (; idx < size ; ++idx) {
        avgValue = avgValue+ valueArray[idx];
        //        printf("%f ",valueArray[idx]);
    }
    
    //    printf("\n");
    
    avgValue = avgValue / ((double)size);
    
    return avgValue;
}

void calMeanVariance(float *valueArray, int size, float *meanValue, float *varianceValue){
    *meanValue = calMean(valueArray, size);
    *varianceValue = 0.0;
    int idx =0;
    
    for (; idx < size; ++idx) {
        *varianceValue = *varianceValue + (valueArray[idx] - *meanValue) * (valueArray[idx] - *meanValue);
    }
    
    *varianceValue = sqrt(*varianceValue)/((double)size);
    
    //    printf("Mean:%f variance:%f\n",*meanValue, *varianceValue);
    return;
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

char randomChar(){
    return rand() % 256;
}

void generateRandomValue(char *data, int size){
    int idx;
    
    srand(time(NULL));
    
    for (idx = 0; idx < size; ++idx) {
        *(data+idx) = randomChar();
    }
}

int measureGCRS(int k, int m, int w,
                 int workSizePerDataParityBlock, int loopNum,
                 float *encodeTimeElapse, float *encodeTimeVariance,
                 float *decodeTimeElapse, float *decodeTimeVariance){
    cudaSetDevice(DEVICE_SELCTED_ID);

    int *bitmatrix;
    unsigned int *encoding_column_bitmatrix, *decoding_column_bitmatrix;

    char *dataPtr, *codePtr, *dataDevPtr, *codeDevPtr;
    char  *inputPtr, *outputPtr, *inputDevPtr, *outputDevPtr;

    float *encodingTimeArray;
    float *decodingTimeArray;
    
    encodingTimeArray = talloc(float, loopNum);
    decodingTimeArray = talloc(float, loopNum);

    if (gcrs_check_k_m_w(k,m,w) < 0) {
        printf("Invalid k,m,w\n");
        return ;
    }
    
    bitmatrix = gcrs_create_bitmatrix(k, m, w);
    //    printMatrix(bitmatrix, k*w, m*w);
    
    if (bitmatrix == NULL) {
        printf("Unable to create bitmatrix\n");
        return ;
    }
    
    int factor = 1;
    if (m*w > (sizeof(unsigned int) * 8)) {
        factor = 2;
    }
    
    int idx;
    
    dataPtr = talloc(char, workSizePerDataParityBlock * k);
    codePtr = talloc(char, workSizePerDataParityBlock * m);
    
    inputPtr = talloc(char, workSizePerDataParityBlock * k);
    outputPtr = talloc(char, workSizePerDataParityBlock * m);
    
    cudaError_t error;
    if((error = cudaMalloc((void **)&dataDevPtr, workSizePerDataParityBlock * k)) != cudaSuccess){
        printf("Encounter error:%s when alloc device memory\n",cudaGetErrorString(error));
        return  -1;
    }
    
    if((error = cudaMalloc((void **)&codeDevPtr, workSizePerDataParityBlock * m)) != cudaSuccess){
        printf("Encounter error:%s when alloc device memory\n",cudaGetErrorString(error));
        return  -1;
    }
    
    if((error = cudaMalloc((void **)&inputDevPtr, workSizePerDataParityBlock * k)) != cudaSuccess){
        printf("Encounter error:%s when alloc device memory\n",cudaGetErrorString(error));
        return  -1;
    }
    
    if((error = cudaMalloc((void **)&outputDevPtr, workSizePerDataParityBlock * m)) != cudaSuccess){
        printf("Encounter error:%s when alloc device memory\n",cudaGetErrorString(error));
        return  -1;
    }
    
    
    generateRandomValue(dataPtr, workSizePerDataParityBlock * k);

    
    int warpThreadNum = 32;
    int threadNum = MAX_THREAD_NUM;
    int workSizePerWarp = warpThreadNum / w * w;
    int workSizePerBlock = threadNum / warpThreadNum * workSizePerWarp * sizeof(long);
    int blockNum = workSizePerDataParityBlock / workSizePerBlock;

    if (workSizePerDataParityBlock % workSizePerBlock != 0) {
        ++blockNum;
    }
    
    int loopIdx;
    for (loopIdx = 0; loopIdx < loopNum; ++loopIdx ){
        encoding_column_bitmatrix = gcrs_create_column_coding_bitmatrix(k, m, w, bitmatrix);
        gcrs_cuda_sm_set_column_coding_bitmatrix(encoding_column_bitmatrix,k, m, w,  factor);
        memset(codePtr, 0, workSizePerDataParityBlock * m);

        cudaMemcpy((void *)dataDevPtr, (void *)dataPtr, workSizePerDataParityBlock * k, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)codeDevPtr, (void *)codePtr, workSizePerDataParityBlock * m, cudaMemcpyHostToDevice);


        gcrs_raw_coding_measure(k, m, w,
                                threadNum, blockNum,
                                workSizePerDataParityBlock / sizeof(long),
                                dataDevPtr, codeDevPtr, (encodingTimeArray + loopIdx));
        
        cudaMemcpy((void *)codePtr, (void *)codeDevPtr, workSizePerDataParityBlock * m, cudaMemcpyDeviceToHost);
        
        int *erasures = talloc(int, k+m+1);
        int erasureNum = randomErasureM(erasures, m, k+m);
        int *erased = gcrs_erasures_to_erased(k, m, erasures);
        int *matIdx = talloc(int, erasureNum);
        int *dm_ids = talloc(int, k);

        int *decoding_bitmatrix = gcrs_create_decoding_bitmatrix(k, m, w, bitmatrix,
                                                                matIdx, erasures, dm_ids);
        if (decoding_bitmatrix == NULL) {
            printf("unable to create decoding bitmatrix\n");
        }
        
        decoding_column_bitmatrix =  gcrs_create_column_coding_bitmatrix(k, m, w, decoding_bitmatrix);
        gcrs_cuda_sm_set_column_coding_bitmatrix(decoding_column_bitmatrix,k, m, w, factor);

        int datablockMissing = 0, kIdx, mIdx;
        
        for (kIdx = 0; kIdx < k; ++kIdx) {
            if (erased[kIdx] == 1) {
                datablockMissing = 1;
            }
        }
        
        //    printf("datablockMissing:%d\n", datablockMissing);
        
        if (datablockMissing == 0) {
            for (kIdx = 0; kIdx < k; ++kIdx) {
                memcpy(inputPtr + kIdx * workSizePerDataParityBlock, dataPtr + kIdx * workSizePerDataParityBlock, workSizePerDataParityBlock);
            }
        }else{
            for (kIdx = 0; kIdx < k; ++kIdx) {
                //            printf("dm_ids[kIdx]:%d\n",dm_ids[kIdx]);
                if (dm_ids[kIdx] < k) {
                    memcpy((inputPtr + kIdx * workSizePerDataParityBlock), (dataPtr + dm_ids[kIdx] * workSizePerDataParityBlock), workSizePerDataParityBlock);
                }else{
                    memcpy((inputPtr+ kIdx * workSizePerDataParityBlock), (codePtr + (dm_ids[kIdx] - k) * workSizePerDataParityBlock), workSizePerDataParityBlock);
                }
            }
        }
        
        memset(outputPtr, 0, workSizePerDataParityBlock);
        cudaMemcpy((void *)inputDevPtr, (void *)inputPtr, workSizePerDataParityBlock * k, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)outputDevPtr, (void *)outputPtr , workSizePerDataParityBlock * m, cudaMemcpyHostToDevice);


        gcrs_raw_coding_measure(k, m, w,
                                threadNum, blockNum,
                                workSizePerDataParityBlock / sizeof(long),
                                inputDevPtr, outputDevPtr, (decodingTimeArray + loopIdx));

        cudaMemcpy((void *)outputPtr, (void *)outputDevPtr, workSizePerDataParityBlock * m, cudaMemcpyDeviceToHost);

        
        // verify
        int sIdx;
        int verifyWorkSize = workSizePerDataParityBlock;
        for (mIdx = 0; mIdx < erasureNum; ++mIdx) {
            char *src_ptr;
            
            if (matIdx[mIdx] < k) {
                src_ptr = dataPtr + matIdx[mIdx] * workSizePerDataParityBlock;
            }else{
                src_ptr = codePtr + (matIdx[mIdx] - k) * workSizePerDataParityBlock;
            }
            
            
            for (sIdx = 0; sIdx < verifyWorkSize ; sIdx = sIdx + 1) {
                if (*(outputPtr + mIdx * workSizePerDataParityBlock + sIdx) != *(src_ptr + sIdx)) {
                    printf("matIdx[mIdx]: %d, %d %d \n",matIdx[mIdx], mIdx, sIdx);
                    break;
                }
            }
        }

        free(erasures);
        free(erased);
        free(matIdx);
        free(dm_ids);
        free(decoding_bitmatrix);
        free(encoding_column_bitmatrix);
        free(decoding_column_bitmatrix);

    }
    
    calMeanVariance(encodingTimeArray, loopNum, encodeTimeElapse, encodeTimeVariance);
    calMeanVariance(decodingTimeArray, loopNum, decodeTimeElapse, decodeTimeVariance);
    
    
    free(encodingTimeArray);
    free(decodingTimeArray);
    
    free(bitmatrix);
    free(dataPtr);
    free(codePtr);
    free(inputPtr);
    free(outputPtr);
    
    cudaFree(dataDevPtr);
    cudaFree(codeDevPtr);
    cudaFree(inputDevPtr);
    cudaFree(outputDevPtr);
    
    cudaDeviceReset();

    return 0;
}
