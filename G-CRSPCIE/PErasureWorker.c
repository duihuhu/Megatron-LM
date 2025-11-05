//
//  PErasureWorker.cpp
//  PErasurePCIE
//
//  Created by Liu Chengjian on 16/10/19.
//  Copyright (c) 2016年 csliu. All rights reserved.
//

#include <pthread.h>

#include "PErasureWorker.h"
#include "PErasureUtilities.h"
#include "GCRSCommon.h"
#include "GCRSCoding.h"
#include "GCRSMatrix.h"

void initTimeRecoder(struct timeRecorder* recorder){
    int idx;
    recorder->hostToDeviceEvent = talloc(cudaEvent_t, recorder->recordSize);
    recorder->deviceToHostEvent = talloc(cudaEvent_t, recorder->recordSize);
    recorder->kernelStartEvent = talloc(cudaEvent_t, recorder->recordSize);
    recorder->kernelEndEvent = talloc(cudaEvent_t, recorder->recordSize);
    
    for (idx = 0; idx < recorder->recordSize; ++idx) {
        cudaEventCreate(&(recorder->hostToDeviceEvent[idx]));
        cudaEventCreate(&(recorder->deviceToHostEvent[idx]));
        cudaEventCreate(&(recorder->kernelStartEvent[idx]));
        cudaEventCreate(&(recorder->kernelEndEvent[idx]));
    }
    cudaEventCreate(&(recorder->endEvent));
    
    recorder->dataToDeviceTimeInMS = talloc(float, recorder->recordSize);
    recorder->codeToHostTimeInMS = talloc(float, recorder->recordSize);
    recorder->kernelTimeInMS = talloc(float, recorder->recordSize);
}

void calculateRecorder(struct timeRecorder* recorder){
    int idx;
    
    cudaEventElapsedTime(&recorder->totalTimeInMS, recorder->hostToDeviceEvent[0], recorder->endEvent);
    
    for (idx = 0; idx != recorder->recordSize; ++idx) {
        if (idx != (recorder->recordSize - 1)) {
            cudaEventElapsedTime(&recorder->dataToDeviceTimeInMS[idx], recorder->hostToDeviceEvent[idx], recorder->hostToDeviceEvent[idx+1]);
            cudaEventElapsedTime(&recorder->codeToHostTimeInMS[idx], recorder->deviceToHostEvent[idx], recorder->deviceToHostEvent[idx+1]);
        }else{
             cudaEventElapsedTime(&recorder->dataToDeviceTimeInMS[idx], recorder->hostToDeviceEvent[idx], recorder->deviceToHostEvent[0]);
             cudaEventElapsedTime(&recorder->codeToHostTimeInMS[idx], recorder->deviceToHostEvent[idx], recorder->endEvent);
        }
        
        cudaEventElapsedTime(&recorder->kernelTimeInMS[idx], recorder->kernelStartEvent[idx], recorder->kernelEndEvent[idx]);
    }
}

void printRecorder(struct timeRecorder* recorder){
    int idx;
    printf("Total time:%fms\n", recorder->totalTimeInMS);
    for (idx = 0; idx != recorder->recordSize; ++idx) {
        printf("Task id:%d, Host to Device:%fms, Device to Host:%fms, Kernel:%fms \n",idx, recorder->dataToDeviceTimeInMS[idx], recorder->codeToHostTimeInMS[idx], recorder->kernelTimeInMS[idx]);
    }
    
}

void deallocRecorder(struct timeRecorder* recorder){
    int idx;
    
    for (idx = 0; idx < recorder->recordSize; ++idx) {
        cudaEventDestroy(recorder->hostToDeviceEvent[idx]);
        cudaEventDestroy(recorder->deviceToHostEvent[idx]);
        cudaEventDestroy(recorder->kernelStartEvent[idx]);
        cudaEventDestroy(recorder->kernelEndEvent[idx]);
    }
    
    cudaEventDestroy(recorder->endEvent);
    
    free(recorder->hostToDeviceEvent);
    free(recorder->deviceToHostEvent);
    free(recorder->kernelStartEvent);
    free(recorder->kernelEndEvent);
    
    free(recorder->dataToDeviceTimeInMS);
    free(recorder->codeToHostTimeInMS);
    free(recorder->kernelTimeInMS);
}

struct PErasureWorker *PErasureWorkerInit(size_t kValue, size_t mValue, size_t wValue, size_t wholeBufSize, size_t taskSize){
    cudaSetDevice(DEVICE_SELCTED_ID);
    
    struct PErasureWorker *worker = (struct PErasureWorker *)malloc(sizeof(struct PErasureWorker));
    if (!worker) return NULL;
    
    worker->k = kValue;
    worker->m = mValue;
    worker->w = wValue;
    worker->bufSize = wholeBufSize;
    worker->taskNum = taskSize;
    worker->dataId = 0;
    worker->kernelId = 0;
    
    initBuf(worker);
    initStreams(worker);
//    initKernelPtr(worker);
    
    // Initialize GCRSMCoding
    GCRSMCodingInit(&worker->mCoding, worker->k, worker->m, worker->w);
    
    // Create and set bitmatrix
    int *bitmatrix = gcrs_create_bitmatrix(worker->k, worker->m, worker->w);
    if (bitmatrix == NULL) {
        printf("Failed to create bitmatrix\n");
        free(worker);
        return NULL;
    }
    GCRSMCodingSetBitmatrix(&worker->mCoding, bitmatrix);
    free(bitmatrix);
    
    worker->recorder.recordSize = taskSize;
    
    initTimeRecoder(&worker->recorder);
    
    return worker;
}

void initBuf(struct PErasureWorker *worker){
    size_t taskIdx, dataSizePerAssign, codeSizePerAssign;
    
//    worker->bufSize = align_value(worker->bufSize, (sizeof(long)*worker->w));
    
    worker->bufSizePerTask = worker->bufSize / worker->taskNum;
    worker->bufSizePerTask = align_value(worker->bufSizePerTask, (sizeof(long)*worker->w));
    worker->bufSize = worker->bufSizePerTask * worker->taskNum;
//    printf("Buf size:%d bufSizePerTask:%d\n", worker->bufSize, worker->bufSizePerTask);
    
    alloc_cuda_host_memory(((void **)&worker->data_host_buf), ((size_t)worker->bufSize*worker->k));
    alloc_cuda_host_memory(((void **)&worker->code_host_buf), ((size_t)worker->bufSize*worker->m));
    alloc_cuda_host_memory(((void **)&worker->input_buf), ((size_t)worker->bufSize*worker->k));
    alloc_cuda_host_memory(((void **)&worker->output_buf), ((size_t)worker->bufSize*worker->m));

    alloc_cuda_device_memory(((void **)&worker->data_dev_buf),((size_t)worker->bufSize*worker->k));
    alloc_cuda_device_memory(((void **)&worker->code_dev_buf), ((size_t)worker->bufSize*worker->m));
    alloc_cuda_device_memory(((void **)&worker->input_dev_buf),((size_t)worker->bufSize*worker->k));
    alloc_cuda_device_memory(((void **)&worker->output_dev_buf), ((size_t)worker->bufSize*worker->m));

    
    generateRandomValue(worker->data_host_buf, worker->bufSize*worker->k);
    
    worker->data_dev_buf_ptr = talloc(char *, worker->taskNum);
    worker->code_dev_buf_ptr = talloc(char *, worker->taskNum);
    
    worker->input_dev_buf_ptr = talloc(char *, worker->taskNum);
    worker->output_dev_buf_ptr = talloc(char *, worker->taskNum);
    worker->external_data_dev_buf = NULL;
    worker->use_external_data_dev_buf = 0;
    worker->skip_d2h_transfer = 0;
    worker->external_code_dev_buf = NULL;
    worker->use_external_code_dev_buf = 0;
    
    // Save internal buffer pointers for proper cleanup
    worker->internal_data_dev_buf = worker->data_dev_buf;
    worker->internal_code_dev_buf = worker->code_dev_buf;
    
    // Initialize CUDA execution configuration
    worker->threads_per_block = MAX_THREAD_NUM;  // Default: 128
    worker->blocks_per_grid = 0;  // 0 means auto-calculate
    worker->use_custom_grid_config = 0;

    dataSizePerAssign = worker->bufSizePerTask * worker->k;
    codeSizePerAssign = worker->bufSizePerTask * worker->m;
    
    for (taskIdx = 0; taskIdx < worker->taskNum; ++taskIdx) {
        worker->data_dev_buf_ptr[taskIdx] = worker->data_dev_buf + dataSizePerAssign * taskIdx;
        worker->code_dev_buf_ptr[taskIdx] = worker->code_dev_buf + codeSizePerAssign * taskIdx;
        worker->input_dev_buf_ptr[taskIdx] = worker->input_dev_buf + dataSizePerAssign * taskIdx;
        worker->output_dev_buf_ptr[taskIdx] = worker->output_dev_buf + codeSizePerAssign * taskIdx;
    }
    
    worker->bufSizeForLastTask = worker->bufSize - (worker->bufSizePerTask * (worker->taskNum - 1));
}

void initKernelPtr(struct PErasureWorker *worker){
    
//    GCRSMCodingInitFuncPtrs(&worker->mCoding);
   // worker->kernel_ptr// = coding_func_array[(worker->m - 1) * (MAX_W - MIN_W + 1)+ mCoding->w - MIN_W];//perasure_sm_coding_ptrs[(worker->m - 1) * (MAX_W - MIN_W + 1)+ worker->w - MIN_W];
//    printf("ptr idx:%lu\n",((worker->m - 1) * (MAX_W - MIN_W + 1)+ worker->w - MIN_W));
}


/**
 *  Step 1: launch a thread for executing kernel and recording kernel execution time.
 *  Step 2: call the function to transfer data from host memory to device memory and trasfer the result from device memory to host memory.
 */
void producerConsumerRun(struct PErasureWorker *worker){
    
    startSingleDirectionDataTransfer(worker);
}

void startSingleDirectionDataTransfer(struct PErasureWorker *worker){
    int warpThreadNum = 32;
    int threadNum = MAX_THREAD_NUM;
    int idx, transferBackId;
    size_t workSizePerWarp = warpThreadNum / worker->w * worker->w;
    size_t workSizePerBlock = threadNum / warpThreadNum * workSizePerWarp * sizeof(long);
    size_t blockNum = worker->bufSizePerTask / workSizePerBlock;
    
    if ((worker->bufSizePerTask % workSizePerBlock) != 0) {
        blockNum = blockNum + 1;
    }

    gettimeofday(&worker->startEncodeTime, NULL);
    do{
//        printf("Transfer host to device:%d\n", worker->dataId);
//        cudaEventRecord(worker->recorder.hostToDeviceEvent[worker->dataId],worker->workerKernelStream[worker->dataId]);
        if (worker->dataId != (worker->taskNum - 1)) {
            transfer_cuda_memory_host_to_device_async((worker->data_dev_buf + worker->dataId * worker->k * worker->bufSizePerTask), (worker->data_host_buf + worker->dataId * worker->k * worker->bufSizePerTask), (worker->k * worker->bufSizePerTask), worker->workerKernelStream[worker->dataId]);
            
//            cudaEventRecord(worker->recorder.kernelStartEvent[worker->dataId],worker->workerKernelStream[worker->dataId]);
            GCRSMCodingCall(&worker->mCoding,
                                worker->data_dev_buf_ptr[worker->dataId], worker->code_dev_buf_ptr[worker->dataId],
                                threadNum,blockNum,
                                worker->bufSizePerTask/sizeof(long),
                                worker->workerKernelStream[worker->dataId]);

//            worker->kernel_ptr(worker->data_dev_buf_ptr[worker->dataId], worker->code_dev_buf_ptr[worker->dataId],
//                               worker->k,
//                               threadNum,blockNum,
//                               worker->bufSizePerTask/sizeof(long),
//                               worker->workerKernelStream[worker->dataId]);
//            cudaEventRecord(worker->recorder.kernelEndEvent[worker->dataId],worker->workerKernelStream[worker->dataId]);

        }else{
            transfer_cuda_memory_host_to_device_async((worker->data_dev_buf + worker->dataId * worker->k * worker->bufSizePerTask), (worker->data_host_buf + worker->dataId * worker->k * worker->bufSizePerTask), (worker->k * worker->bufSizeForLastTask),worker->workerKernelStream[worker->dataId]);
//            cudaThreadSynchronize();

//            cudaEventRecord(worker->recorder.kernelStartEvent[worker->dataId],worker->workerKernelStream[worker->dataId]);
            transferBackId  = 0;
            do {
                
                //            printf("Transfer device to host:%d\n", transferBackId);
//                cudaEventRecord(worker->recorder.deviceToHostEvent[transferBackId],worker->workerKernelStream[worker->dataId-1]);
                
                if (transferBackId != (worker->taskNum -1)) {
                    transfer_cuda_memory_device_to_host_async((worker->code_host_buf + transferBackId * worker->m * worker->bufSizePerTask), (worker->code_dev_buf+transferBackId * worker->m * worker->bufSizePerTask), (worker->m * worker->bufSizePerTask),worker->workerKernelStream[worker->dataId-1]);
                }
                
                ++transferBackId;
            } while (transferBackId != worker->taskNum-1);
            
            GCRSMCodingCall(&worker->mCoding,
                                worker->data_dev_buf_ptr[worker->dataId], worker->code_dev_buf_ptr[worker->dataId],
                                threadNum,blockNum,
                                worker->bufSizeForLastTask/sizeof(long),
                                worker->workerKernelStream[worker->dataId]);
            
            transfer_cuda_memory_device_to_host_async((worker->code_host_buf + transferBackId * worker->m * worker->bufSizePerTask), (worker->code_dev_buf + transferBackId * worker->m * worker->bufSizePerTask), (worker->m * worker->bufSizeForLastTask),worker->workerKernelStream[worker->dataId]);


//            cudaEventRecord(worker->recorder.kernelEndEvent[worker->dataId],worker->workerKernelStream[worker->dataId]);

        }
        
        ++worker->dataId;
    }while (worker->dataId != worker->taskNum);
    
    cudaDeviceSynchronize();
    gettimeofday(&worker->endEncodeTime, NULL);
//    cudaEventRecord(worker->recorder.endEvent,worker->workerKernelStream[worker->dataId-1]);
//    
//    for (idx = 0 ; idx < worker->taskNum; ++idx) {
//        cudaEventSynchronize(worker->recorder.hostToDeviceEvent[idx]);
//        cudaEventSynchronize(worker->recorder.kernelEndEvent[idx]);
//        cudaEventSynchronize(worker->recorder.deviceToHostEvent[idx]);
//    }
    
//    cudaEventSynchronize(worker->recorder.endEvent);
}

void fullDuplexRunEncode(struct PErasureWorker *worker){
    
    int warpThreadNum = 32;
    int threadNum = worker->threads_per_block;  // Use configurable threads_per_block
    int idx;
    size_t workSizePerWarp = warpThreadNum / worker->w * worker->w;
    size_t workSizePerBlock = threadNum / warpThreadNum * workSizePerWarp * sizeof(long);
    size_t blockNum;
    
    // Calculate blockNum: use custom if set, otherwise auto-calculate
    if (worker->use_custom_grid_config && worker->blocks_per_grid > 0) {
        blockNum = worker->blocks_per_grid;
    } else {
        blockNum = worker->bufSizePerTask / workSizePerBlock;
        if ((worker->bufSizePerTask % workSizePerBlock) != 0) {
            blockNum = blockNum + 1;
        }
    }
    
    gettimeofday(&worker->startEncodeTime, NULL);
    for (idx = 0; idx < worker->taskNum; ++idx) {
        if (idx != worker->taskNum-1) {
            
            if (worker->use_external_data_dev_buf && worker->external_data_dev_buf != NULL) {
                // External buffer is already set as data_dev_buf in PErasureWorkerSetInputDevicePtr
                // data_dev_buf_ptr already points to the correct locations, no need to copy
                // No H2D transfer here; external buffer is assumed already on device
            } else {
                transfer_cuda_memory_host_to_device_async((worker->data_dev_buf + idx * worker->k * worker->bufSizePerTask), (worker->data_host_buf + idx * worker->k * worker->bufSizePerTask), (worker->k * worker->bufSizePerTask),worker->workerKernelStream[idx]);
            }
            
//            worker->kernel_ptr(worker->data_dev_buf_ptr[idx], worker->code_dev_buf_ptr[idx],
//                               worker->k,
//                               threadNum,blockNum,
//                               worker->bufSizePerTask/sizeof(long),
//                               worker->workerKernelStream[idx]);

            GCRSMCodingCall(&worker->mCoding,
                                worker->data_dev_buf_ptr[idx], worker->code_dev_buf_ptr[idx],
                                threadNum,blockNum,
                                worker->bufSizePerTask/sizeof(long),
                                worker->workerKernelStream[idx]);

            // Debugging: check for CUDA errors after kernel launch
            {
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    fprintf(stderr, "CUDA error after GCRSMCodingCall (task %d): %s\n", idx, cudaGetErrorString(err));
                }
                // synchronize to force error reporting to this location
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    fprintf(stderr, "cudaDeviceSynchronize error after GCRSMCodingCall (task %d): %s\n", idx, cudaGetErrorString(err));
                }
            }

            // Skip D2H transfer if zero-copy GPU tensor encoding is enabled
            if (!worker->skip_d2h_transfer) {
                transfer_cuda_memory_device_to_host_async((worker->code_host_buf + idx * worker->m * worker->bufSizePerTask), (worker->code_dev_buf + idx * worker->m * worker->bufSizePerTask), (worker->m * worker->bufSizePerTask),worker->workerKernelStream[idx]);
            }
        }else{
            if (worker->use_external_data_dev_buf && worker->external_data_dev_buf != NULL) {
                memcpy((worker->data_dev_buf + idx * worker->k * worker->bufSizePerTask), (worker->external_data_dev_buf + idx * worker->k * worker->bufSizePerTask), 0);
            } else {
                transfer_cuda_memory_host_to_device_async((worker->data_dev_buf + idx * worker->k * worker->bufSizePerTask), (worker->data_host_buf + idx * worker->k * worker->bufSizePerTask), (worker->k * worker->bufSizeForLastTask),worker->workerKernelStream[idx]);
            }

//            worker->kernel_ptr(worker->data_dev_buf_ptr[idx], worker->code_dev_buf_ptr[idx],
//                               worker->k,
//                               threadNum,blockNum,
//                               worker->bufSizeForLastTask/sizeof(long),
//                               worker->workerKernelStream[idx]);

            GCRSMCodingCall(&worker->mCoding,
                            worker->data_dev_buf_ptr[idx], worker->code_dev_buf_ptr[idx],
                            threadNum,blockNum,
                            worker->bufSizeForLastTask/sizeof(long),
                            worker->workerKernelStream[idx]);
            
            // Debugging: check for CUDA errors after kernel launch for last task
            {
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    fprintf(stderr, "CUDA error after GCRSMCodingCall (last task %d): %s\n", idx, cudaGetErrorString(err));
                }
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    fprintf(stderr, "cudaDeviceSynchronize error after GCRSMCodingCall (last task %d): %s\n", idx, cudaGetErrorString(err));
                }
            }

            // Skip D2H transfer if zero-copy GPU tensor encoding is enabled
            if (!worker->skip_d2h_transfer) {
                transfer_cuda_memory_device_to_host_async((worker->code_host_buf + idx * worker->m * worker->bufSizePerTask), (worker->code_dev_buf + idx * worker->m * worker->bufSizePerTask), (worker->m * worker->bufSizeForLastTask),worker->workerKernelStream[idx]);
            }

        }

    }
    
    cudaDeviceSynchronize();
    gettimeofday(&worker->endEncodeTime, NULL);
    
}

void fullDuplexRunDecode(struct PErasureWorker *worker){
    
    int warpThreadNum = 32;
    int threadNum = worker->threads_per_block;  // Use configurable threads_per_block
    int idx;
    size_t workSizePerWarp = warpThreadNum / worker->w * worker->w;
    size_t workSizePerBlock = threadNum / warpThreadNum * workSizePerWarp * sizeof(long);
    size_t blockNum;
    
    // Calculate blockNum: use custom if set, otherwise auto-calculate
    if (worker->use_custom_grid_config && worker->blocks_per_grid > 0) {
        blockNum = worker->blocks_per_grid;
    } else {
        blockNum = worker->bufSizePerTask / workSizePerBlock;
        if ((worker->bufSizePerTask % workSizePerBlock) != 0) {
            blockNum = blockNum + 1;
        }
    }
    
    gettimeofday(&worker->startDecodeTime, NULL);
    for (idx = 0; idx < worker->taskNum; ++idx) {
        if (idx != worker->taskNum-1) {
            
            transfer_cuda_memory_host_to_device_async((worker->input_dev_buf + idx * worker->k * worker->bufSizePerTask), (worker->input_buf + idx * worker->k * worker->bufSizePerTask), (worker->k * worker->bufSizePerTask),worker->workerKernelStream[idx]);
            GCRSMCodingCall(&worker->mCoding,
                            worker->input_dev_buf_ptr[idx], worker->output_dev_buf_ptr[idx],
                            threadNum,blockNum,
                            worker->bufSizePerTask/sizeof(long),
                            worker->workerKernelStream[idx]);
//            worker->kernel_ptr(worker->input_dev_buf_ptr[idx], worker->output_dev_buf_ptr[idx],
//                               worker->k,
//                               threadNum,blockNum,
//                               worker->bufSizePerTask/sizeof(long),
//                               worker->workerKernelStream[idx]);
            // Skip D2H transfer if zero-copy GPU tensor decoding is enabled
            if (!worker->skip_d2h_transfer) {
                transfer_cuda_memory_device_to_host_async((worker->output_buf + idx * worker->m * worker->bufSizePerTask), (worker->output_dev_buf + idx * worker->m * worker->bufSizePerTask), (worker->m * worker->bufSizePerTask),worker->workerKernelStream[idx]);
            }
        }else{
            transfer_cuda_memory_host_to_device_async((worker->input_dev_buf + idx * worker->k * worker->bufSizePerTask), (worker->input_buf + idx * worker->k * worker->bufSizePerTask), (worker->k * worker->bufSizeForLastTask),worker->workerKernelStream[idx]);
            
//            worker->kernel_ptr(worker->input_dev_buf_ptr[idx], worker->output_dev_buf_ptr[idx],
//                               worker->k,
//                               threadNum,blockNum,
//                               worker->bufSizeForLastTask/sizeof(long),
//                               worker->workerKernelStream[idx]);
            GCRSMCodingCall(&worker->mCoding,
                            worker->input_dev_buf_ptr[idx], worker->output_dev_buf_ptr[idx],
                            threadNum,blockNum,
                            worker->bufSizeForLastTask/sizeof(long),
                            worker->workerKernelStream[idx]);

            // Skip D2H transfer if zero-copy GPU tensor decoding is enabled
            if (!worker->skip_d2h_transfer) {
                transfer_cuda_memory_device_to_host_async((worker->output_buf + idx * worker->m * worker->bufSizePerTask), (worker->output_dev_buf + idx * worker->m * worker->bufSizePerTask), (worker->m * worker->bufSizeForLastTask),worker->workerKernelStream[idx]);
            }
            
        }
        
    }
    
    cudaDeviceSynchronize();
    gettimeofday(&worker->endDecodeTime, NULL);
    
}

void setupDecoding(struct PErasureWorker *worker, int *dm_ids){
    int idxTaskNum = 0, idx;
    
    for (idxTaskNum = 0; idxTaskNum < worker->taskNum; ++idxTaskNum) {
        size_t sizePerBlock = 0;
        if (idxTaskNum < worker->taskNum-1) {
            sizePerBlock = worker->bufSizePerTask;
        }else{
            sizePerBlock = worker->bufSizeForLastTask;
        }

        for (idx = 0; idx < worker->k; ++idx) {

            if (dm_ids[idx] < worker->k) {
                memcpy((worker->input_buf + idxTaskNum * worker->k * worker->bufSizePerTask + idx*sizePerBlock), (worker->data_host_buf + idxTaskNum * worker->k * worker->bufSizePerTask + dm_ids[idx] * sizePerBlock), sizePerBlock);
                
            }else{
                memcpy((worker->input_buf + idxTaskNum * worker->k * worker->bufSizePerTask + idx*sizePerBlock), (worker->code_host_buf + idxTaskNum * worker->m * worker->bufSizePerTask + (dm_ids[idx]-worker->k) * sizePerBlock), sizePerBlock);
            }
        }

    }
}

void checkEncodeDecodeResult(struct PErasureWorker *worker, int *matIdx, int erasureNum){
    int idxTaskNum = 0, idx, verifyIdx;
    
//    for (idx = 0; idx < erasureNum; ++idx) {
//        printf("matIdx[%d]:%d ",idx, matIdx[idx]);
//    }
//    printf("\n");
    
    for (idxTaskNum = 0; idxTaskNum < worker->taskNum; ++idxTaskNum) {
        size_t sizePerBlock = 0;
        if (idxTaskNum < worker->taskNum-1) {
            sizePerBlock = worker->bufSizePerTask;
        }else{
            sizePerBlock = worker->bufSizeForLastTask;
        }
        
        for (idx = 0; idx < erasureNum; ++idx) {
//            printf("matIdx[%d]:%d\n",idx, matIdx[idx]);
            char *src_ptr, *des_ptr;
            if (matIdx[idx] < worker->k) {
                src_ptr = worker->data_host_buf + idxTaskNum * worker->k * worker->bufSizePerTask + matIdx[idx] * sizePerBlock;
            }else{
                src_ptr = worker->code_host_buf + idxTaskNum * worker->m * worker->bufSizePerTask + (matIdx[idx] - worker->k)* sizePerBlock;
            }
            
            des_ptr = worker->output_buf + idxTaskNum * worker->m * worker->bufSizePerTask + idx * sizePerBlock;
            
            for (verifyIdx = 0; verifyIdx < sizePerBlock; ++verifyIdx) {
                if (*(src_ptr + verifyIdx) != *(des_ptr +verifyIdx)) {
                    printf("sizePerBlock:%d, last size per block:%d, TaskIdx:%d, erasure idx:%d, offset:%d not equal!\n", sizePerBlock, worker->bufSizeForLastTask, idxTaskNum, matIdx[idx], verifyIdx);
                    break;
                }
            }
        }
    }
}

void PErasureWorkerCalculateRecord(struct PErasureWorker *worker){
//    calculateRecorder(&worker->recorder);
    worker->encodeTimeConsume = get_elapsed_time_in_ms(worker->startEncodeTime, worker->endEncodeTime);
    worker->decodeTimeConsume = get_elapsed_time_in_ms(worker->startDecodeTime, worker->endDecodeTime);
}

void PErasureWorkerPrintRecord(struct PErasureWorker *worker){
//    printRecorder(&worker->recorder);
    printf("Time consume:%fms\n", worker->encodeTimeConsume);
}

double PErasureWorkerGetEncodeTimeConsume(struct PErasureWorker *worker){
    return worker->encodeTimeConsume;
}

double PErasureWorkerGetDecodeTimeConsume(struct PErasureWorker *worker){
    return worker->decodeTimeConsume;
}


void PErasureWorkerDealloc(struct PErasureWorker *worker){
    int idx;

    free_cuda_host_memory(worker->data_host_buf);
    free_cuda_host_memory(worker->code_host_buf);
    free_cuda_host_memory(worker->input_buf);
    free_cuda_host_memory(worker->output_buf);

    // Only free internal buffers, not external ones
    if (!worker->use_external_data_dev_buf && worker->internal_data_dev_buf != NULL) {
        free_cuda_device_memory(worker->internal_data_dev_buf);
    }
    if (!worker->use_external_code_dev_buf && worker->internal_code_dev_buf != NULL) {
        free_cuda_device_memory(worker->internal_code_dev_buf);
    }
    free_cuda_device_memory(worker->input_dev_buf);
    free_cuda_device_memory(worker->output_dev_buf);

    
    free(worker->data_dev_buf_ptr);
    free(worker->code_dev_buf_ptr);
    free(worker->input_dev_buf_ptr);
    free(worker->output_dev_buf_ptr);

    
    for (idx = 0 ; idx < worker->taskNum; ++idx) {
        destroy_cuda_stream(&(worker->workerKernelStream[idx]));
    }
    
    deallocRecorder(&worker->recorder);

    // NOTE: Do not reset CUDA device here unconditionally; provide explicit function
    // to reset device when desired to avoid interfering with external CUDA contexts
    // cudaDeviceReset();
}

void initStreams(struct PErasureWorker *worker){
    // one task, one stream;
    int idx;
    worker->workerKernelStream = talloc(cudaStream_t, worker->taskNum);
    for (idx = 0 ; idx < worker->taskNum; ++idx) {
        create_cuda_stream(&(worker->workerKernelStream[idx]));
    }
}

void PErasureWorkerSetInputData(struct PErasureWorker *worker, char *input_data, size_t data_size) {
    if (data_size > worker->bufSize * worker->k) {
        printf("Input data size %zu exceeds buffer size %zu\n", data_size, worker->bufSize * worker->k);
        return;
    }
    memcpy(worker->data_host_buf, input_data, data_size);
}

// New API: set a device pointer to external device buffer. If set, the worker
// will use this device buffer directly and skip copying host->device for input.
void PErasureWorkerSetInputDevicePtr(struct PErasureWorker *worker, char *device_ptr, size_t data_size) {
    // Note: data_size validation omitted here; caller must ensure buffer is large enough
    worker->external_data_dev_buf = device_ptr;
    worker->use_external_data_dev_buf = 1;
    // Replace internal device buffer pointer to point to the external buffer
    // Keep internal buffer pointer for cleanup
    worker->data_dev_buf = device_ptr;
    // Recompute per-task device pointers
    size_t dataSizePerAssign = worker->bufSizePerTask * worker->k;
    for (size_t taskIdx = 0; taskIdx < worker->taskNum; ++taskIdx) {
        worker->data_dev_buf_ptr[taskIdx] = worker->data_dev_buf + dataSizePerAssign * taskIdx;
    }
}

void PErasureWorkerGetOutputData(struct PErasureWorker *worker, char *output_data, size_t data_size) {
    if (data_size > worker->bufSize * worker->m) {
        printf("Output data size %zu exceeds buffer size %zu\n", data_size, worker->bufSize * worker->m);
        return;
    }
    memcpy(output_data, worker->code_host_buf, data_size);
}

// New API: get output device pointer for zero-copy GPU output
char *PErasureWorkerGetOutputDevicePtr(struct PErasureWorker *worker) {
    return worker->code_dev_buf;
}

// New API to enable/disable D2H transfer (for zero-copy GPU tensor encoding)
void PErasureWorkerSetSkipD2HTransfer(struct PErasureWorker *worker, int skip) {
    worker->skip_d2h_transfer = skip;
}

// New API to set external output device buffer (for zero-copy GPU tensor output)
void PErasureWorkerSetOutputDevicePtr(struct PErasureWorker *worker, char *device_ptr, size_t data_size) {
    // Note: data_size validation omitted here; caller must ensure buffer is large enough
    worker->external_code_dev_buf = device_ptr;
    worker->use_external_code_dev_buf = 1;
    // Replace internal device buffer pointer to point to the external buffer
    // Keep internal buffer pointer for cleanup
    worker->code_dev_buf = device_ptr;
    // Recompute per-task device pointers
    size_t codeSizePerAssign = worker->bufSizePerTask * worker->m;
    for (size_t taskIdx = 0; taskIdx < worker->taskNum; ++taskIdx) {
        worker->code_dev_buf_ptr[taskIdx] = worker->code_dev_buf + codeSizePerAssign * taskIdx;
    }
}

// New API for GPU tensor zero-copy encoding: input and output are both on GPU
// This function sets up both input and output device pointers and enables zero-copy mode
void PErasureWorkerEncodeGPUZeroCopy(struct PErasureWorker *worker, char *input_dev_ptr, char *output_dev_ptr, size_t data_size) {
    // Set input device pointer (skip H2D)
    PErasureWorkerSetInputDevicePtr(worker, input_dev_ptr, data_size);
    // Set output device pointer (skip D2H)
    PErasureWorkerSetOutputDevicePtr(worker, output_dev_ptr, data_size);
    // Enable skip D2H transfer
    PErasureWorkerSetSkipD2HTransfer(worker, 1);
}

void PErasureWorkerResetDevice(){
    cudaDeviceReset();
}

// New API to configure CUDA execution parameters
void PErasureWorkerSetThreadsPerBlock(struct PErasureWorker *worker, int threads) {
    if (threads <= 0 || threads > 1024) {
        printf("Warning: threads_per_block should be between 1 and 1024, using default %d\n", MAX_THREAD_NUM);
        worker->threads_per_block = MAX_THREAD_NUM;
    } else {
        // Ensure threads is a multiple of warp size (32) for best performance
        if (threads % 32 != 0) {
            printf("Warning: threads_per_block (%d) is not a multiple of 32, rounding up\n", threads);
            threads = ((threads + 31) / 32) * 32;
        }
        worker->threads_per_block = threads;
    }
}

void PErasureWorkerSetBlocksPerGrid(struct PErasureWorker *worker, int blocks) {
    if (blocks < 0) {
        printf("Warning: blocks_per_grid cannot be negative, disabling custom config\n");
        worker->use_custom_grid_config = 0;
        worker->blocks_per_grid = 0;
    } else if (blocks == 0) {
        // 0 means auto-calculate
        worker->use_custom_grid_config = 0;
        worker->blocks_per_grid = 0;
    } else {
        worker->use_custom_grid_config = 1;
        worker->blocks_per_grid = blocks;
    }
}

int PErasureWorkerGetThreadsPerBlock(struct PErasureWorker *worker) {
    return worker->threads_per_block;
}

int PErasureWorkerGetBlocksPerGrid(struct PErasureWorker *worker) {
    if (worker->use_custom_grid_config) {
        return worker->blocks_per_grid;
    } else {
        // Calculate and return auto-calculated value
        int warpThreadNum = 32;
        int threadNum = worker->threads_per_block;
        size_t workSizePerWarp = warpThreadNum / worker->w * worker->w;
        size_t workSizePerBlock = threadNum / warpThreadNum * workSizePerWarp * sizeof(long);
        int blockNum = worker->bufSizePerTask / workSizePerBlock;
        if ((worker->bufSizePerTask % workSizePerBlock) != 0) {
            blockNum = blockNum + 1;
        }
        return blockNum;
    }
}