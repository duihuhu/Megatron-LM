//
//  main.c
//  NoBitmatrixOpt
//
//  Created by Liu Chengjian on 17/8/28.
//  Copyright (c) 2017年 csliu. All rights reserved.
//

#include <stdio.h>
#include "MeasureGCRS.h"
#include "GCRSCommon.h"

#define ARG_SIZE 5

void Usage(){
    printf("Input:./Program m minBlockNum maxBlockNum loopSize\n");
    exit(0);
}

int main(int argc, const char * argv[]) {
    // insert code here...
    int k, m, w;
    double preTimeElapse;
    int minBlockNum, maxBlockNum;
    int workSizePerDataParityBlock, alignedWorkSizePerDataParityBlock, loopSize;
    float encodingTimeAvg, encodingTimeVariance, decodingTimeAvg, decodingTimeVariance;
    
    if (argc < ARG_SIZE) {
        Usage();
    }
    
    k = 2;
    m = atoi(argv[1]);
    minBlockNum = atoi(argv[2]);
    maxBlockNum = atoi(argv[3]);
    loopSize = atoi(argv[4]);
    
    int blocksNum, blockIdx = 1;
    printf("BlockId\t w4\t w5 \t w6 \t w7 \t w8\n");
    
for (blocksNum = minBlockNum; blocksNum <= maxBlockNum; blocksNum = blocksNum * 2) {
    printf("%d",blockIdx);

    for (w = 4; w <= 8; ++w) {
        //        w = 4;
        //        w = perasure_check_k_m_w(k, m, w);
        
        if (w == -1) {
            printf("w = -1");
            exit(0);
        }
        
        int sizePerBlock = (THREADS_PER_BLOCK / w) * w * sizeof(long);

        alignedWorkSizePerDataParityBlock = sizePerBlock * blocksNum;
        
        if (alignedWorkSizePerDataParityBlock % (w*sizeof(long)) != 0) {
            alignedWorkSizePerDataParityBlock = alignedWorkSizePerDataParityBlock + ((w*sizeof(long)) - alignedWorkSizePerDataParityBlock % (w*sizeof(long)));
        }
        
        measureGCRS(k, m, w,
                    alignedWorkSizePerDataParityBlock, loopSize,
                    &encodingTimeAvg, &encodingTimeVariance,
                    &decodingTimeAvg, &decodingTimeVariance);
        float dataSize = (k)*alignedWorkSizePerDataParityBlock;
        float encodeThroughput = dataSize/encodingTimeAvg*1000.0/1024.0/1024.0/1024.0;
        float decodeThroughput = dataSize/decodingTimeAvg*1000.0/1024.0/1024.0/1024.0;
        preTimeElapse = encodingTimeAvg;
        
//        printf("%d \t %f \t %f \t %f \t %f \t %f\n",
//               w, encodingTimeAvg, encodeThroughput, decodingTimeAvg, decodeThroughput, (dataSize/1024.0/1024.0));
        printf("\t%f",encodeThroughput);

    }
    
    printf("\n");

    ++blockIdx;
}
    return 0;
}
