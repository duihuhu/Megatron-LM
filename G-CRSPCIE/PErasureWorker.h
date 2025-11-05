//
//  PErasureWorker.h
//  PErasurePCIE
//
//  Created by Liu Chengjian on 16/10/19.
//  Copyright (c) 2016年 csliu. All rights reserved.
//

#ifndef __PErasurePCIE__PErasureWorker__
#define __PErasurePCIE__PErasureWorker__

#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "GCRSCoding.h"
#include "PErasureUtilities.h"

struct timeRecorder{
    size_t recordSize;
    cudaEvent_t  *hostToDeviceEvent;
    cudaEvent_t *kernelStartEvent;
    cudaEvent_t *kernelEndEvent;
    cudaEvent_t *deviceToHostEvent;
    cudaEvent_t endEvent;
    
    float totalTimeInMS;
    float *dataToDeviceTimeInMS;
    float *codeToHostTimeInMS;
    float *kernelTimeInMS;
};

void initTimeRecoder(struct timeRecorder* recorder);
void calculateRecorder(struct timeRecorder* recorder);

struct PErasureWorker{
//    void (*kernel_ptr)(char *dataPtr, char *codeDevPtr,
//                       int k,
//                       int threadDimX,int blockDimX,
//                       int workSizePerGridInLong,
//                       cudaStream_t stream);
    
    int dataId;
    int kernelId;
    
    size_t k;
    size_t m;
    size_t w;
    
    size_t bufSize;
    size_t taskNum;
    size_t bufSizePerTask;
    size_t bufSizeForLastTask;
    char *data_host_buf;
    char *code_host_buf;
    
    char *input_buf;
    char *output_buf;
    
    char *data_dev_buf;
    char *code_dev_buf;
    
    char *input_dev_buf;
    char *output_dev_buf;
    
    char **data_dev_buf_ptr;
    char **code_dev_buf_ptr;
    
    char **input_dev_buf_ptr;
    char **output_dev_buf_ptr;
    /* If external device buffer provided by caller, point here and skip H2D */
    char *external_data_dev_buf;
    int use_external_data_dev_buf;
    
    /* Internal device buffers (saved for proper cleanup) */
    char *internal_data_dev_buf;
    char *internal_code_dev_buf;
    
    /* If set, skip device-to-host transfer for zero-copy GPU tensor encoding */
    int skip_d2h_transfer;
    
    /* External output device buffer (optional, for zero-copy output) */
    char *external_code_dev_buf;
    int use_external_code_dev_buf;
    
    struct timeRecorder recorder;
    struct GCRSMCoding mCoding;
    cudaStream_t *workerKernelStream;
    
    struct timeval startEncodeTime;
    struct timeval endEncodeTime;
    
    struct timeval endDecodeTime;
    struct timeval startDecodeTime;

    double encodeTimeConsume;
    double decodeTimeConsume;
    
    /* CUDA execution configuration (configurable) */
    int threads_per_block;  // Number of threads per block (default: MAX_THREAD_NUM)
    int blocks_per_grid;    // Number of blocks per grid (auto-calculated if 0, or manually set)
    int use_custom_grid_config;  // If 1, use custom blocks_per_grid instead of auto-calculating
};

void producerConsumerRun(struct PErasureWorker *worker);
void startSingleDirectionDataTransfer(struct PErasureWorker *worker);

void fullDuplexRunEncode(struct PErasureWorker *worker);

void setupDecoding(struct PErasureWorker *worker, int *dm_ids);
void fullDuplexRunDecode(struct PErasureWorker *worker);
void checkEncodeDecodeResult(struct PErasureWorker *worker, int *matIdx, int erasureNum);

struct PErasureWorker *PErasureWorkerInit(size_t kValue, size_t mValue, size_t wValue, size_t wholeBufSize, size_t taskSize);

void PErasureWorkerSetInputData(struct PErasureWorker *worker, char *input_data, size_t data_size);

// New API to set a device pointer as input buffer (zero-copy path)
void PErasureWorkerSetInputDevicePtr(struct PErasureWorker *worker, char *device_ptr, size_t data_size);

void PErasureWorkerGetOutputData(struct PErasureWorker *worker, char *output_data, size_t data_size);

// New API to get output device pointer (zero-copy path for output)
char *PErasureWorkerGetOutputDevicePtr(struct PErasureWorker *worker);

// New API to enable/disable D2H transfer (for zero-copy GPU tensor encoding)
void PErasureWorkerSetSkipD2HTransfer(struct PErasureWorker *worker, int skip);

// New API to set external output device buffer (for zero-copy GPU tensor output)
void PErasureWorkerSetOutputDevicePtr(struct PErasureWorker *worker, char *device_ptr, size_t data_size);

// New API for GPU tensor zero-copy encoding: input and output are both on GPU
void PErasureWorkerEncodeGPUZeroCopy(struct PErasureWorker *worker, char *input_dev_ptr, char *output_dev_ptr, size_t data_size);

// New API to configure CUDA execution parameters
void PErasureWorkerSetThreadsPerBlock(struct PErasureWorker *worker, int threads);
void PErasureWorkerSetBlocksPerGrid(struct PErasureWorker *worker, int blocks);
int PErasureWorkerGetThreadsPerBlock(struct PErasureWorker *worker);
int PErasureWorkerGetBlocksPerGrid(struct PErasureWorker *worker);

void PErasureWorkerResetDevice();

void PErasureWorkerCalculateRecord(struct PErasureWorker *worker);
void PErasureWorkerPrintRecord(struct PErasureWorker *worker);
void PErasureWorkerDealloc(struct PErasureWorker *worker);

void initBuf(struct PErasureWorker *worker);
void initStreams(struct PErasureWorker *worker);
void initKernelPtr(struct PErasureWorker *worker);

double PErasureWorkerGetEncodeTimeConsume(struct PErasureWorker *worker);
double PErasureWorkerGetDecodeTimeConsume(struct PErasureWorker *worker);
#endif /* defined(__PErasurePCIE__PErasureWorker__) */
