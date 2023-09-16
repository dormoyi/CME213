#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"



////////////////////////////////////////////////////////////////////////////////
//                         COMBINED KERNELS                                   //
////////////////////////////////////////////////////////////////////////////////

/////                         GEMM SIGMOID REPMAT                           /////


template<int numYPerStep>
__global__
void algorithm3_sigmoid_repmat(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D) {

    int thread_rank = threadIdx.x + threadIdx.y * blockDim.x;
    int row = thread_rank + blockDim.x * blockDim.y * blockIdx.x;
    int col = blockIdx.y*blockDim.x;

    nn_real A_loc[4] = { 0 };
    nn_real res[16] = { 0 };

    __shared__ nn_real blockB[numYPerStep][16+1];

    for (int bloc_id = 0; bloc_id < (K + numYPerStep - 1) / numYPerStep; bloc_id++){
        // load block B into shared memory, each thread reads one element
        if (threadIdx.y + numYPerStep*bloc_id < K && col + threadIdx.x < N) 
            blockB[threadIdx.y][threadIdx.x] = B[threadIdx.y + numYPerStep*bloc_id + (col + threadIdx.x)*K];
        else
            blockB[threadIdx.y][threadIdx.x] = 0;

        
        __syncthreads();


        // get A in local memory
        //int step_iter = bloc_id == (K + numYPerStep - 1) / numYPerStep -1 && K%numYPerStep!=0 ? K%numYPerStep : numYPerStep;
        if (row < M){
            for (int step =0; step<numYPerStep; step++){
                if (numYPerStep*bloc_id + step < K)
                    A_loc[step] = A[row + M*(step + numYPerStep*bloc_id)];
            }
        }


        for (int c = 0; c < blockDim.x; c++){
            for (int step = 0; step<numYPerStep; step++){
                // if (col+c < N && row < M && (bloc_id*numYPerStep + step) < K)
                    res[c] += A_loc[step] * blockB[step][c]; 
            }
        }

        __syncthreads();
    }

    // beta*C
    for (int c = 0; c < blockDim.x; c++){
        if (row < M && col + c < N){
            // tot = beta * C[row + (c+col)*M] + alpha * res[c]; 
            D[row + (c+col)*M] =  1.0 / ( 1.0 + exp(- (beta * C[row] + alpha * res[c]) ));
        }
    }
}

int algorithm3_sigmoid_repmat_caller(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D)
{
    int xthread = 16;
    int ythread = 4; 
    dim3 threads(xthread, ythread);

    const int numYPerStep = 4;
    int xblock = (M + threads.x * numYPerStep - 1)/ threads.x / numYPerStep; 
    int yblock = (N + threads.y * numYPerStep - 1)/threads.y / numYPerStep;
    dim3 blocks(xblock, yblock);

    algorithm3_sigmoid_repmat<numYPerStep><<<blocks,threads>>>(A, B, C, alpha, beta, M, N, K, D);

    return 0;

}


template<int side>
__global__
void algorithm2_sigmoid_repmat(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    nn_real mult_result = 0;

    __shared__ nn_real blockA[side+1][side+1]; 
    __shared__ nn_real blockB[side+1][side+1];

    for (int bloc_id = 0; bloc_id < (K + side - 1) / side; bloc_id++){
        // load into shared memory
        if (i < M && threadIdx.y + side*bloc_id < K)
            blockA[threadIdx.x][threadIdx.y] = A[i + M*(threadIdx.y + side*bloc_id)];
        if (threadIdx.x + side*bloc_id < K && j < N)
            blockB[threadIdx.x][threadIdx.y] = B[(threadIdx.x + side*bloc_id) + j*K];
        
        __syncthreads();

        // perform the computation for the bloc
        if (i < M && j < N) {
            // A*B
            int iter = bloc_id == (K + side - 1) / side -1 && K%side!=0 ? K%side : side;
            for (int k_bloc = 0; k_bloc < iter; k_bloc++)
                mult_result += blockA[threadIdx.x][k_bloc] * blockB[k_bloc][threadIdx.y];
        }

        __syncthreads();
    }

    if (i < M && j < N) {
        // beta*C
        D[i + j*M] = 1.0 / ( 1.0 + exp(- (beta * C[i] + alpha * mult_result)  ));  
    }
}


int algorithm2_sigmoid_repmat_caller(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D)
{

    int xthread = 16;
    int ythread = 16; 
    dim3 threads(xthread, ythread);

    const int side = 16;
    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    algorithm2_sigmoid_repmat<side><<<blocks,threads>>>(A, B, C, alpha, beta, M, N, K, D);

    return 0;
    
}


int myGEMMSigmoidRepmat(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D)
{
    // A M x K
    // B K x N
    // C M x N

    // algorithm2_sigmoid_repmat_caller(A, B, C, alpha, beta, M, N, K, D);
    algorithm3_sigmoid_repmat_caller(A, B, C, alpha, beta, M, N, K, D);
    return 0;
}

/////                         GEMM        REPMAT                           /////

template<int numYPerStep>
__global__
void algorithm3_repmat(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D) {

    int thread_rank = threadIdx.x + threadIdx.y * blockDim.x;
    int row = thread_rank + blockDim.x * blockDim.y * blockIdx.x;
    int col = blockIdx.y*blockDim.x;

    nn_real A_loc[4] = { 0 };
    nn_real res[16] = { 0 };

    __shared__ nn_real blockB[numYPerStep][16+1];

    for (int bloc_id = 0; bloc_id < (K + numYPerStep - 1) / numYPerStep; bloc_id++){
        // load block B into shared memory, each thread reads one element
        if (threadIdx.y + numYPerStep*bloc_id < K && col + threadIdx.x < N) 
            blockB[threadIdx.y][threadIdx.x] = B[threadIdx.y + numYPerStep*bloc_id + (col + threadIdx.x)*K];
        else
            blockB[threadIdx.y][threadIdx.x] = 0;

        
        __syncthreads();


        // get A in local memory
        //int step_iter = bloc_id == (K + numYPerStep - 1) / numYPerStep -1 && K%numYPerStep!=0 ? K%numYPerStep : numYPerStep;
        if (row < M){
            for (int step =0; step<numYPerStep; step++){
                if (numYPerStep*bloc_id + step < K)
                    A_loc[step] = A[row + M*(step + numYPerStep*bloc_id)];
            }
        }


        for (int c = 0; c < blockDim.x; c++){
            for (int step = 0; step<numYPerStep; step++){
                // if (col+c < N && row < M && (bloc_id*numYPerStep + step) < K)
                    res[c] += A_loc[step] * blockB[step][c]; 
            }
        }

        __syncthreads();
    }

    // beta*C
    for (int c = 0; c < blockDim.x; c++){
        if (row < M && col + c < N){
            // tot = beta * C[row + (c+col)*M] + alpha * res[c]; 
            D[row + (c+col)*M] = (beta * C[row] + alpha * res[c]) ;
        }
    }
}

int algorithm3_repmat_caller(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D)
{
    int xthread = 16;
    int ythread = 4; 
    dim3 threads(xthread, ythread);

    const int numYPerStep = 4;
    int xblock = (M + threads.x * numYPerStep - 1)/ threads.x / numYPerStep; 
    int yblock = (N + threads.y * numYPerStep - 1)/threads.y / numYPerStep;
    dim3 blocks(xblock, yblock);

    algorithm3_repmat<numYPerStep><<<blocks,threads>>>(A, B, C, alpha, beta, M, N, K, D);

    return 0;

}


int myGEMMRepmat(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D)
{
    // A M x K
    // B K x N
    // C M x N

    // algorithm2_caller(A, B, C, alpha, beta, M, N, K);
    algorithm3_repmat_caller(A, B, C, alpha, beta, M, N, K, D);
    return 0;
}






/////                         GEMM SIGMOID                                 /////


template<int numYPerStep>
__global__
void algorithm3_sigmoid(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K) {

    int thread_rank = threadIdx.x + threadIdx.y * blockDim.x;
    int row = thread_rank + blockDim.x * blockDim.y * blockIdx.x;
    int col = blockIdx.y*blockDim.x;

    nn_real A_loc[4] = { 0 };
    nn_real res[16] = { 0 };
    nn_real tot;

    __shared__ nn_real blockB[numYPerStep][16+1];

    for (int bloc_id = 0; bloc_id < (K + numYPerStep - 1) / numYPerStep; bloc_id++){
        // load block B into shared memory, each thread reads one element
        if (threadIdx.y + numYPerStep*bloc_id < K && col + threadIdx.x < N) 
            blockB[threadIdx.y][threadIdx.x] = B[threadIdx.y + numYPerStep*bloc_id + (col + threadIdx.x)*K];
        else
            blockB[threadIdx.y][threadIdx.x] = 0;

        
        __syncthreads();


        // get A in local memory
        //int step_iter = bloc_id == (K + numYPerStep - 1) / numYPerStep -1 && K%numYPerStep!=0 ? K%numYPerStep : numYPerStep;
        if (row < M){
            for (int step =0; step<numYPerStep; step++){
                if (numYPerStep*bloc_id + step < K)
                    A_loc[step] = A[row + M*(step + numYPerStep*bloc_id)];
            }
        }


        for (int c = 0; c < blockDim.x; c++){
            for (int step = 0; step<numYPerStep; step++){
                // if (col+c < N && row < M && (bloc_id*numYPerStep + step) < K)
                    res[c] += A_loc[step] * blockB[step][c]; 
            }
        }

        __syncthreads();
    }

    // beta*C
    for (int c = 0; c < blockDim.x; c++){
        if (row < M && col + c < N){
            tot = beta * C[row + (c+col)*M] + alpha * res[c]; 
            C[row + (c+col)*M] =  1.0 / ( 1.0 + exp(- tot  ));
        }
    }
}

int algorithm3_sigmoid_caller(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K)
{
    int xthread = 16;
    int ythread = 4; 
    dim3 threads(xthread, ythread);

    const int numYPerStep = 4;
    int xblock = (M + threads.x * numYPerStep - 1)/ threads.x / numYPerStep; 
    int yblock = (N + threads.y * numYPerStep - 1)/threads.y / numYPerStep;
    dim3 blocks(xblock, yblock);

    algorithm3_sigmoid<numYPerStep><<<blocks,threads>>>(A, B, C, alpha, beta, M, N, K);

    return 0;

}

template<int side>
__global__
void algorithm2_sigmoid(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    nn_real mult_result = 0;

    __shared__ nn_real blockA[side+1][side+1]; 
    __shared__ nn_real blockB[side+1][side+1];

    for (int bloc_id = 0; bloc_id < (K + side - 1) / side; bloc_id++){
        // load into shared memory
        if (i < M && threadIdx.y + side*bloc_id < K)
            blockA[threadIdx.x][threadIdx.y] = A[i + M*(threadIdx.y + side*bloc_id)];
        if (threadIdx.x + side*bloc_id < K && j < N)
            blockB[threadIdx.x][threadIdx.y] = B[(threadIdx.x + side*bloc_id) + j*K];
        
        __syncthreads();

        // perform the computation for the bloc
        if (i < M && j < N) {
            // A*B
            int iter = bloc_id == (K + side - 1) / side -1 && K%side!=0 ? K%side : side;
            for (int k_bloc = 0; k_bloc < iter; k_bloc++)
                mult_result += blockA[threadIdx.x][k_bloc] * blockB[k_bloc][threadIdx.y];
        }

        __syncthreads();
    }

    if (i < M && j < N) {
        // beta*C
        C[i + j*M] = 1.0 / ( 1.0 + exp(- (beta * C[i + j*M] + alpha * mult_result)  ));  
    }
}

int algorithm2_sigmoid_caller(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K)
{

    int xthread = 16;
    int ythread = 16; 
    dim3 threads(xthread, ythread);

    const int side = 16;
    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    algorithm2_sigmoid<side><<<blocks,threads>>>(A, B, C, alpha, beta, M, N, K);

    return 0;
    
}


int myGEMMSigmoid(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K)
{
    // A M x K
    // B K x N
    // C M x N

    // algorithm2_sigmoid_caller(A, B, C, alpha, beta, M, N, K);
    algorithm3_sigmoid_caller(A, B, C, alpha, beta, M, N, K);
    return 0;
}



/////                         GEMM TRANSPOSE                              /////


template<int numYPerStep>
__global__
void algorithm3_transpose(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K) {

    int thread_rank = threadIdx.x + threadIdx.y * blockDim.x;
    int row = thread_rank + blockDim.x * blockDim.y * blockIdx.x;
    int col = blockIdx.y*blockDim.x;

    nn_real A_loc[4] = { 0 };
    nn_real res[16] = { 0 };

    __shared__ nn_real blockB[numYPerStep][16+1];

    for (int bloc_id = 0; bloc_id < (K + numYPerStep - 1) / numYPerStep; bloc_id++){
        // load block B into shared memory, each thread reads one element
        if (threadIdx.y + numYPerStep*bloc_id < K && col + threadIdx.x < N) 
            blockB[threadIdx.y][threadIdx.x] = B[threadIdx.y + numYPerStep*bloc_id + (col + threadIdx.x)*K];
        else
            blockB[threadIdx.y][threadIdx.x] = 0;

        
        __syncthreads();


        // get A in local memory
        if (row < M){
            for (int step =0; step<numYPerStep; step++){
                if (step + numYPerStep*bloc_id < K)
                    A_loc[step] = A[row*K + (step + numYPerStep*bloc_id)];
            }
        }

        // perform computation
        for (int c = 0; c < blockDim.x; c++){
            for (int step = 0; step<numYPerStep; step++){
                    res[c] += A_loc[step] * blockB[step][c]; 
            }
        }

        __syncthreads();
    }

    // beta*C
    for (int c = 0; c < blockDim.x; c++){
        if (row < M && col + c < N)
            C[row + (c+col)*M] = beta * C[row + (c+col)*M] + alpha * res[c]; 
    }
}



int algorithm3_transpose_caller(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K)
{
    int xthread = 16;
    int ythread = 4; 
    dim3 threads(xthread, ythread);

    const int numYPerStep = 4;
    int xblock = (M + threads.x * numYPerStep - 1)/ threads.x / numYPerStep; 
    int yblock = (N + threads.y * numYPerStep - 1)/threads.y / numYPerStep;
    dim3 blocks(xblock, yblock);

    algorithm3_transpose<numYPerStep><<<blocks,threads>>>(A, B, C, alpha, beta, M, N, K);



    return 0;

}


int myGEMMTranspose(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K)
{
    // A M x K
    // B K x N
    // C M x N

    // algorithm2_caller(A, B, C, alpha, beta, M, N, K);
    algorithm3_transpose_caller(A, B, C, alpha, beta, M, N, K);




    return 0;
}



/////                        SCALAR DIFFERENCE                             /////


__global__
void scalarDifferenceKernel(nn_real *A, nn_real *B, int M, int N, nn_real alpha) {
    // A: src, B: dest

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < M && j < N) {
        A[i+M*j] = alpha* (A[i+M*j] - B[i+M*j]);
    }
}

int myScalarDifference(nn_real *A, nn_real *B, int M, int N, nn_real alpha){
    // A - B, dst = C, dimensions MxN

    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    scalarDifferenceKernel<<<blocks,threads>>>(A, B, M, N, alpha);

    return 0;
}


/////                        LR     DIFFERENCE                             /////


__global__
void LRDifferenceKernel(nn_real *A, nn_real *B, int M, int N, nn_real alpha) {
    // A: src, B: dest

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < M && j < N) {
        A[i+M*j] =  A[i+M*j] - alpha * B[i+M*j];
        B[i+M*j] = A[i+M*j];
    }
}

int myLRDifference(nn_real *A, nn_real *B, int M, int N, nn_real alpha){
    // A - B, dst = C, dimensions MxN

    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    LRDifferenceKernel<<<blocks,threads>>>(A, B, M, N, alpha);

    return 0;
}

/////                   GEMM SUM ROW                                      /////

template<int numYPerStep>
__global__
void algorithm3_sumRow(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D) {

    int thread_rank = threadIdx.x + threadIdx.y * blockDim.x;
    int row = thread_rank + blockDim.x * blockDim.y * blockIdx.x;
    int col = blockIdx.y*blockDim.x;

    nn_real A_loc[4] = { 0 };
    nn_real res[16] = { 0 };

    D[row] = 0;

    __shared__ nn_real blockB[numYPerStep][16+1];

    for (int bloc_id = 0; bloc_id < (K + numYPerStep - 1) / numYPerStep; bloc_id++){
        // load block B into shared memory, each thread reads one element
        if (threadIdx.y + numYPerStep*bloc_id < K && col + threadIdx.x < N) 
            blockB[threadIdx.y][threadIdx.x] = B[threadIdx.y + numYPerStep*bloc_id + (col + threadIdx.x)*K];
        else
            blockB[threadIdx.y][threadIdx.x] = 0;

        
        __syncthreads();


        // get A in local memory
        //int step_iter = bloc_id == (K + numYPerStep - 1) / numYPerStep -1 && K%numYPerStep!=0 ? K%numYPerStep : numYPerStep;
        if (row < M){
            for (int step =0; step<numYPerStep; step++){
                if (numYPerStep*bloc_id + step < K){
                    A_loc[step] = A[row + M*(step + numYPerStep*bloc_id)];
                    D[row] += A_loc[step];
                }
            }
        }


        for (int c = 0; c < blockDim.x; c++){
            for (int step = 0; step<numYPerStep; step++){
                // if (col+c < N && row < M && (bloc_id*numYPerStep + step) < K)
                    res[c] += A_loc[step] * blockB[step][c]; 
            }
        }

        __syncthreads();
    }

    // beta*C
    for (int c = 0; c < blockDim.x; c++){
        if (row < M && col + c < N){
            C[row + (c+col)*M] = beta * C[row + (c+col)*M] + alpha * res[c]; 
        }
    }
}

int algorithm3_sumRow_caller(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D)
{
    int xthread = 16;
    int ythread = 4; 
    dim3 threads(xthread, ythread);

    const int numYPerStep = 4;
    int xblock = (M + threads.x * numYPerStep - 1)/ threads.x / numYPerStep; 
    int yblock = (N + threads.y * numYPerStep - 1)/threads.y / numYPerStep;
    dim3 blocks(xblock, yblock);

    algorithm3_sumRow<numYPerStep><<<blocks,threads>>>(A, B, C, alpha, beta, M, N, K, D);

    return 0;

}

int myGEMMSumRow(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D)
{
    // A M x K
    // B K x N
    // C M x N

    // algorithm2_caller(A, B, C, alpha, beta, M, N, K);
    algorithm3_sumRow_caller(A, B, C, alpha, beta, M, N, K, D);
    return 0;
}



/////                   GEMM TRANSPOSE SUM ROW                             /////

template<int numYPerStep>
__global__
void algorithm3_sumRow_transpose(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D) {

    int thread_rank = threadIdx.x + threadIdx.y * blockDim.x;
    int row = thread_rank + blockDim.x * blockDim.y * blockIdx.x;
    int col = blockIdx.y*blockDim.x;

    nn_real A_loc[4] = { 0 };
    nn_real res[16] = { 0 };

    D[row] = 0;

    __shared__ nn_real blockB[numYPerStep][16+1];

    for (int bloc_id = 0; bloc_id < (K + numYPerStep - 1) / numYPerStep; bloc_id++){
        // load block B into shared memory, each thread reads one element
        if (threadIdx.y + numYPerStep*bloc_id < K && col + threadIdx.x < N) 
            blockB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + numYPerStep*bloc_id)*N + (col + threadIdx.x)];
        else
            blockB[threadIdx.y][threadIdx.x] = 0;


            // if (row < M){
            // for (int step =0; step<numYPerStep; step++){
            //     if (step + numYPerStep*bloc_id < K)
            //         A_loc[step] = A[row*K + (step + numYPerStep*bloc_id)];
            // }

        
        __syncthreads();


        // get A in local memory
        if (row < M){
            for (int step =0; step<numYPerStep; step++){
                if (numYPerStep*bloc_id + step < K){
                    A_loc[step] = A[row + M*(step + numYPerStep*bloc_id)];
                    D[row] += A_loc[step];
                }
            }
        }


        for (int c = 0; c < blockDim.x; c++){
            for (int step = 0; step<numYPerStep; step++){
                // if (col+c < N && row < M && (bloc_id*numYPerStep + step) < K)
                    res[c] += A_loc[step] * blockB[step][c]; 
            }
        }

        __syncthreads();
    }

    // beta*C
    for (int c = 0; c < blockDim.x; c++){
        if (row < M && col + c < N){
            C[row + (c+col)*M] = beta * C[row + (c+col)*M] + alpha * res[c]; 
        }
    }
}

int algorithm3_sumRow_transpose_caller(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D)
{
    int xthread = 16;
    int ythread = 4; 
    dim3 threads(xthread, ythread);

    const int numYPerStep = 4;
    int xblock = (M + threads.x * numYPerStep - 1)/ threads.x / numYPerStep; 
    int yblock = (N + threads.y * numYPerStep - 1)/threads.y / numYPerStep;
    dim3 blocks(xblock, yblock);

    algorithm3_sumRow_transpose<numYPerStep><<<blocks,threads>>>(A, B, C, alpha, beta, M, N, K, D);

    return 0;

}

int myGEMMSumRowTranspose(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K, nn_real *__restrict__ D)
{
    // A M x K
    // B K x N
    // C M x N

    // algorithm2_caller(A, B, C, alpha, beta, M, N, K);
    algorithm3_sumRow_transpose_caller(A, B, C, alpha, beta, M, N, K, D);
    return 0;
}


/////                  ONE MINUS HADAMARD                                  /////



__global__
void hadamardKernel2(nn_real *d_gradz1, nn_real *d_a1, nn_real *d_grada1, int M, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < M && j < N)
        d_gradz1[i+M*j] = d_grada1[i+M*j] * (1 - d_a1[i+M*j]) * d_a1[i+M*j];
}

int myOneMinusHadamard(nn_real *A, nn_real *B, nn_real *C, int M, int N){
    // A % B, dst = A, dimensions MxN

    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    hadamardKernel2<<<blocks,threads>>>(A, B, C, M, N);

    return 0;
}




////////////////////////////////////////////////////////////////////////////////
//                             GEMM KERNELS                                   //
////////////////////////////////////////////////////////////////////////////////


__global__
void gpuMatrix1D(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K) {

    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;

    int i = idx_x % M;
    int j = idx_x / M;

    if (idx_x < M*N) {
        C[i+M*j] *= beta;
        for(int k=0; k<K ; k++){
            C[i+j*M] += alpha*A[i+M*k]*B[k+K*j];
        }
    }

}

__global__
void gpuMatrixBlock(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    nn_real sum = 0;
    if (i < M && j < N) {
        // beta*C
        C[i + j*M] = beta * C[i + j*M];

        // alpha*A*B
        for (int k = 0; k < K; k++){
            sum += A[i + k*M] * B[k + j*K];
        }
        C[i + j*M] += alpha * sum; 
    }
}

template<int side>
__global__
void algorithm2(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    nn_real mult_result = 0;

    __shared__ nn_real blockA[side+1][side+1]; 
    __shared__ nn_real blockB[side+1][side+1];

    for (int bloc_id = 0; bloc_id < (K + side - 1) / side; bloc_id++){
        // load into shared memory
        if (i < M && threadIdx.y + side*bloc_id < K)
            blockA[threadIdx.x][threadIdx.y] = A[i + M*(threadIdx.y + side*bloc_id)];
        if (threadIdx.x + side*bloc_id < K && j < N)
            blockB[threadIdx.x][threadIdx.y] = B[(threadIdx.x + side*bloc_id) + j*K];
        
        __syncthreads();

        // perform the computation for the bloc
        if (i < M && j < N) {
            // A*B
            int iter = bloc_id == (K + side - 1) / side -1 && K%side!=0 ? K%side : side;
            for (int k_bloc = 0; k_bloc < iter; k_bloc++)
                mult_result += blockA[threadIdx.x][k_bloc] * blockB[k_bloc][threadIdx.y];
        }

        __syncthreads();
    }

    if (i < M && j < N) {
        // beta*C
        C[i + j*M] = beta * C[i + j*M] + alpha * mult_result; 
    }
}


template<int numYPerStep>
__global__
void algorithm3(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K) {

    int thread_rank = threadIdx.x + threadIdx.y * blockDim.x;
    int row = thread_rank + blockDim.x * blockDim.y * blockIdx.x;
    int col = blockIdx.y*blockDim.x;

    nn_real A_loc[4] = { 0 };
    nn_real res[16] = { 0 };

    __shared__ nn_real blockB[numYPerStep][16+1];

    for (int bloc_id = 0; bloc_id < (K + numYPerStep - 1) / numYPerStep; bloc_id++){
        // load block B into shared memory, each thread reads one element
        if (threadIdx.y + numYPerStep*bloc_id < K && col + threadIdx.x < N) 
            blockB[threadIdx.y][threadIdx.x] = B[threadIdx.y + numYPerStep*bloc_id + (col + threadIdx.x)*K];
        else
            blockB[threadIdx.y][threadIdx.x] = 0;

        
        __syncthreads();


        // get A in local memory
        //int step_iter = bloc_id == (K + numYPerStep - 1) / numYPerStep -1 && K%numYPerStep!=0 ? K%numYPerStep : numYPerStep;
        if (row < M){
            for (int step =0; step<numYPerStep; step++){
                if (numYPerStep*bloc_id + step < K)
                    A_loc[step] = A[row + M*(step + numYPerStep*bloc_id)];
            }
        }


        for (int c = 0; c < blockDim.x; c++){
            for (int step = 0; step<numYPerStep; step++){
                // if (col+c < N && row < M && (bloc_id*numYPerStep + step) < K)
                    res[c] += A_loc[step] * blockB[step][c]; 
            }
        }

        __syncthreads();
    }

    // beta*C
    for (int c = 0; c < blockDim.x; c++){
        if (row < M && col + c < N)
            C[row + (c+col)*M] = beta * C[row + (c+col)*M] + alpha * res[c]; 
    }
}

int algorithm3_caller(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K)
{
    int xthread = 16;
    int ythread = 4; 
    dim3 threads(xthread, ythread);

    const int numYPerStep = 4;
    int xblock = (M + threads.x * numYPerStep - 1)/ threads.x / numYPerStep; 
    int yblock = (N + threads.y * numYPerStep - 1)/threads.y / numYPerStep;
    dim3 blocks(xblock, yblock);

    algorithm3<numYPerStep><<<blocks,threads>>>(A, B, C, alpha, beta, M, N, K);

    return 0;

}

int algorithm2_caller(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K)
{

    int xthread = 16;
    int ythread = 16; 
    dim3 threads(xthread, ythread);

    const int side = 16;
    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    algorithm2<side><<<blocks,threads>>>(A, B, C, alpha, beta, M, N, K);

    return 0;
    
}


int myGEMM(nn_real *__restrict__ A, nn_real *__restrict__ B,
            nn_real *__restrict__ C,
            nn_real alpha, nn_real beta,
            int M, int N, int K)
{
    // A M x K
    // B K x N
    // C M x N

    // algorithm2_caller(A, B, C, alpha, beta, M, N, K);
    algorithm3_caller(A, B, C, alpha, beta, M, N, K);
    return 0;
}


////////////////////////////////////////////////////////////////////////////////
//                            OTHER KERNELS                                   //
////////////////////////////////////////////////////////////////////////////////

__global__
void sigmoidKernel(nn_real *__restrict__ A, int M, int N) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;


    if ((i < M) && (j < N))
    A[i+M*j] = 1.0 / ( 1.0 + exp( -A[i+M*j] ) );
}

int mySigmoid(nn_real *__restrict__ A, int M, int N){

    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    sigmoidKernel<<<blocks,threads>>>(A, M, N);

    return 0;

}

__global__
void expKernel(nn_real *__restrict__ A, int M, int N) {
    // softmax on each column of the matrix

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    A[i+M*j] = exp(A[i+M*j]);
}


__global__
void softmaxKernel(nn_real *__restrict__ A, int M, int N) {
    // softmax on each column of the matrix

    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // int j = threadIdx.y + blockIdx.y * blockDim.y;

    // nn_real sum = 0;
    // if (i < M && j < N) {
    // for (int k=0; k<M; k++){
    //     sum += A[k+M*j];
    // }

    // A[i+M*j] = A[i+M*j] / sum;
    // }

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    nn_real sum = 0;
    if (i < M && j < N) {
    for (int k=0; k<M; k++){
        sum += exp(A[k+M*j]);
    }

    A[i+M*j] = exp(A[i+M*j]) / sum;
    }
}

int mySoftmax(nn_real *__restrict__ A, int M, int N){
    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    // expKernel<<<blocks,threads>>>(A, M, N);
    // cudaDeviceSynchronize();
    softmaxKernel<<<blocks,threads>>>(A, M, N);

    return 0;

}

__global__
void repeatKernelRow(nn_real *__restrict__ A, nn_real *__restrict__ B, int M, int N) {
    // A: dest, B: src, M: size of B, N: desired size
    // M=1, repeat the line

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < M && j < N) {
        A[i+M*j] = B[j];
    }
}

__global__
void repeatKernelCol(nn_real *__restrict__ A, nn_real *__restrict__ B, int M, int N) {
    // A: dest, B: src, M: size of B, N: desired size
    // N=1, repeat the column

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < M && j < N) {
        A[i+M*j] = B[i];
    }
}

int myRepeatCol(nn_real *__restrict__ A, nn_real *__restrict__ B, int M, int N){
    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    repeatKernelCol<<<blocks,threads>>>(A, B, M, N);

    return 0;
}

int myRepeatRow(nn_real *__restrict__ A, nn_real *__restrict__ B, int M, int N){
    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    repeatKernelRow<<<blocks,threads>>>(A, B, M, N);

    return 0;
}


__global__
void transposeKernel(nn_real *A, nn_real *B, int M, int N) {
    // A: src, B: dest

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < M && j < N) {
        B[j+N*i] = A[i+M*j];
    }
}

int myTranspose(nn_real *A, nn_real *B, int M, int N){
    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    transposeKernel<<<blocks,threads>>>(A, B, M, N);


    return 0;
}



__global__
void differenceKernel(nn_real *A, nn_real *B, nn_real *C, int M, int N) {
    // A: src, B: dest

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < M && j < N) {
        C[i+M*j] = A[i+M*j] - B[i+M*j];
    }
}

int myDifference(nn_real *A, nn_real *B, nn_real *C, int M, int N){
    // A - B, dst = C, dimensions MxN

    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    differenceKernel<<<blocks,threads>>>(A, B, C, M, N);

    return 0;
}



__global__
void hadamardKernel(nn_real *A, nn_real *B, int M, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < M && j < N) {
        A[i+M*j] = A[i+M*j] * B[i+M*j];
    }
}

int myHadamardProduct(nn_real *A, nn_real *B, int M, int N){
    // A % B, dst = A, dimensions MxN

    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    hadamardKernel<<<blocks,threads>>>(A, B, M, N);

    return 0;
}

// mySumCol(nn.d_diffy, nn.d_gradb2, nn.d_H2, nn.d_N, 1);


__global__
void SumRowKernel(nn_real *A, nn_real *B, int M, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < M) {
        B[i] = 0;
        for (int k = 0; k < N; k++)
            B[i] += A[i+M*k];
    }
}

int mySumRow(nn_real *A, nn_real *B, int M, int N){
    // sum each row of A, dst = B, dimensions MxN

    int xthread = 32;
    int ythread = 1; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = 1; ////
    dim3 blocks(xblock, yblock);

    SumRowKernel<<<blocks,threads>>>(A, B, M, N);

    return 0;
}



// myOneMinus(nn.dsigdx, nn.d_a1, nn.d_H1, nn.d_N);


__global__
void oneMinusKernel(nn_real *A, nn_real *B, int M, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < M && j < N) {
        B[i + M*j] = 1 - A[i + M*j];
    }
}

int myOneMinus(nn_real *A, nn_real *B, int M, int N){
    // sum each row of A, dst = B, dimensions MxN

    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    oneMinusKernel<<<blocks,threads>>>(A, B, M, N);

    return 0;
}



__global__
void scalarMultKernel(nn_real *A, nn_real alpha, int M, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < M && j < N) {
        A[i + M*j] = alpha * A[i + M*j];
    }
}

int myScalarMult(nn_real *A, nn_real alpha, int M, int N){
    // sum each row of A, dst = B, dimensions MxN

    int xthread = 32;
    int ythread = 32; 
    dim3 threads(xthread, ythread);

    int xblock = (M + threads.x - 1)/threads.x; 
    int yblock = (N + threads.y - 1)/threads.y; 
    dim3 blocks(xblock, yblock);

    scalarMultKernel<<<blocks,threads>>>(A, alpha, M, N);

    return 0;
}