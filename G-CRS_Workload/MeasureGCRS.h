//
//  MeasureGCRS.h
//  NoBitmatrixOpt
//
//  Created by Liu Chengjian on 17/8/28.
//  Copyright (c) 2017年 csliu. All rights reserved.
//

#ifndef __NoBitmatrixOpt__MeasureGCRS__
#define __NoBitmatrixOpt__MeasureGCRS__

#include <stdio.h>

int measureGCRS(int k, int m, int w,
                 int workSizePerDataParityBlock, int loopNum,
                 float *encodeTimeElapse, float *encodeTimeVariance,
                 float *decodeTimeElapse, float *decodeTimeVariance);

#endif /* defined(__NoBitmatrixOpt__MeasureGCRS__) */
