#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utils/common.h"
#include "utils/gpu_util.h"

int myGEMM(nn_real *A, nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
           int M, int N, int K);

int myGEMMSigmoid(nn_real *__restrict__ A, nn_real *__restrict__ B,
nn_real *__restrict__ C,
nn_real alpha, nn_real beta,
int M, int N, int K);

int myGEMMSumRow(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D);

int myGEMMTranspose(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K);

int myGEMMRepmat(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D);

int myGEMMSigmoidRepmat(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D);


int myGEMMSumRowTranspose(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D);

int myLRDifference(nn_real *A, nn_real *B, int M, int N, nn_real alpha);

int myOneMinusHadamard(nn_real *A, nn_real *B, nn_real *C, int M, int N);

int myScalarDifference(nn_real *A, nn_real *B, int M, int N,nn_real alpha);

int mySigmoid(nn_real *__restrict__ A, int M, int N);

int mySoftmax(nn_real *__restrict__ A, int M, int N);

int myRepeatRow(nn_real *__restrict__ A, nn_real *__restrict__ B, int M, int N);

int myRepeatCol(nn_real *__restrict__ A, nn_real *__restrict__ B, int M, int N);

int myTranspose(nn_real *__restrict__ A, nn_real *__restrict__ B, int M, int N);

int myDifference(nn_real *A, nn_real *B, nn_real *C, int M, int N);

int myHadamardProduct(nn_real *A, nn_real *B, int M, int N);
int mySumRow(nn_real *A, nn_real *B, int M, int N);


int myOneMinus(nn_real *A, nn_real *B, int M, int N);

int myScalarMult(nn_real *A, nn_real alpha, int M, int N); 


#endif
