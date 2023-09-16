#include "tests.h"

#include <chrono>
#include <fstream>
#include <iomanip>

#include "../gpu_func.h"
#include "common.h"
#include "cublas_v2.h"
#include "mpi.h"
using namespace std;

#define NUM_ITERS 4 // Number of GEMMs run for timing purposes

#ifdef USE_DOUBLE
#define SCALE 4   // Factor to SCALE the GEMM problem size by
#define TOL 1e-14 // Tolerance for tests
#else
#define SCALE 10 // Factor to SCALE the GEMM problem size by
#define TOL 2e-6 // Tolerance for tests
#endif

// check whether the matrix from Seq is the same as from Par.
// write out mismatches to a file.
int checkErrors(const arma::Mat<nn_real> &Seq, const arma::Mat<nn_real> &Par,
                std::ofstream &ofs, std::vector<nn_real> &errors)
{
  int error = 0;

  for (int i = 0; i < Seq.n_rows; ++i)
  {
    for (int j = 0; j < Seq.n_cols; ++j)
    {
      if (abs(Seq(i, j) - Par(i, j)) > TOL)
      {
        ofs << "Mismatch at pos (" << i << ", " << j
            << ") diff: " << Seq(i, j) - Par(i, j) << " seq: " << Seq(i, j)
            << " par: " << Par(i, j) << endl;
        ++error;
      }
    }
  }

  if (error)
  {
    ofs << "There were " << error
        << " total locations where there was a difference between the seq and "
           "par"
        << endl;
  }
  else
  {
    ofs << "No errors were found" << endl;
  }

  nn_real err_max = arma::norm(Seq - Par, "inf") / arma::norm(Seq, "inf");
  nn_real err_l2 = arma::norm(Seq - Par, 2) / arma::norm(Seq, 2);

  if (err_max > TOL * 1e2)
  {
    cout << "Correctness test failed" << endl;
  }

  errors.push_back(err_max);
  errors.push_back(err_l2);

  return error;
}

int checkNNErrors(NeuralNetwork &seq_nn, NeuralNetwork &par_nn,
                  std::string filename)
{
  std::vector<nn_real> errors_w, errors_b;
  int error = 0;
  std::ofstream ofs(filename.c_str());
  if (!ofs.good())
  {
    std::cerr << "Unable to open the file " << filename << std::endl;
    exit(1);
  }

  for (int i = 0; i < seq_nn.num_layers; i++)
  {
    ofs << "Mismatches for W[" << i << "]" << endl;
    error += checkErrors(seq_nn.W[i], par_nn.W[i], ofs, errors_w);
    ofs << "Mismatches for b[" << i << "]" << endl;
    error += checkErrors(seq_nn.b[i], par_nn.b[i], ofs, errors_b);

    // Writing to file
    ofs << "Max norm of diff b/w seq and par: W[" << i
        << "]: " << setprecision(6) << errors_w[2 * i] << ", b[" << i
        << "]: " << errors_b[2 * i] << endl;
    ofs << "l2  norm of diff b/w seq and par: W[" << i
        << "]: " << setprecision(6) << errors_w[2 * i + 1] << ", b[" << i
        << "]: " << errors_b[2 * i + 1] << endl;

    // Writing to standard output
    cout << "Max norm of diff b/w seq and par: W[" << i
         << "]: " << setprecision(6) << errors_w[2 * i] << ", b[" << i
         << "]: " << errors_b[2 * i] << endl;
    cout << "l2  norm of diff b/w seq and par: W[" << i
         << "]: " << setprecision(6) << errors_w[2 * i + 1] << ", b[" << i
         << "]: " << errors_b[2 * i + 1] << endl;
  }

  ofs.close();
  return error;
}

////////////////////////////////////////////////////////////////////////////////
//                      CREATE MATRICES                                       //
////////////////////////////////////////////////////////////////////////////////

void createMATS(nn_real *A, nn_real *B, nn_real *C1, nn_real *C2, int NI,
                int NJ, int NK)
{
  int i, j;

  for (j = 0; j < NK; j++)
  {
    for (i = 0; i < NI; i++)
    {
      A[i + j * NI] = ((nn_real)i * j) / NI;
    }
  }

  for (j = 0; j < NJ; j++)
  {
    for (i = 0; i < NK; i++)
    {
      B[i + j * NK] = ((nn_real)i * j + 1) / NJ;
    }
  }

  for (j = 0; j < NJ; j++)
  {
    for (i = 0; i < NI; i++)
    {
      C1[i + j * NI] = 0;
      C2[i + j * NI] = ((nn_real)i * j + 2) / NJ;
    }
  }
}

void createMAT(nn_real *A, int M, int N)
{
  int i, j;

  for (j = 0; j < N; j++)
  {
    for (i = 0; i < M; i++)
    {
      A[i + j * M] = ((nn_real)(i+1)*j + 2) / (1000 * M);
    }
  }
}

void createMAT2(nn_real *A, int M, int N)
{
  int i, j;
  int fac;

  for (j = 0; j < N; j++)
  {
    for (i = 0; i < M; i++)
    {
      A[i + j * M] = ((nn_real)i* j + 1.0) / (nn_real)M;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//                              GEMM TEST                                     //
////////////////////////////////////////////////////////////////////////////////

int compareGEMMResults(nn_real *myC, nn_real *refC, int NI, int NJ)
{
  int i, j;
  int fail = 0;

  arma::Mat<nn_real> mysol = arma::Mat<nn_real>(myC, NI, NJ, false);
  arma::Mat<nn_real> refsol = arma::Mat<nn_real>(refC, NI, NJ, false);

  nn_real reldiff =
      arma::norm(mysol - refsol, "inf") / arma::norm(refsol, "inf");

  if (reldiff > TOL)
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "My GEMM output not matching with reference. Rel diff = "
              << reldiff << std::endl;
  }
  else
  {
    std::cout << "GEMM matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }

  return fail;
}

void TestGEMM(int M, int N, int K)
{
  nn_real *A;
  nn_real *B;
  nn_real *C1;
  nn_real *C2;

  nn_real *dA;
  nn_real *dB;
  nn_real *dC1;
  nn_real *dC2;
  nn_real *dummy;

  nn_real alpha = 2.0;
  nn_real beta = 5.0;

  int num_iters = 100;

  A = (nn_real *)malloc(M * K * sizeof(nn_real));
  B = (nn_real *)malloc(K * N * sizeof(nn_real));
  C1 = (nn_real *)malloc(M * N * sizeof(nn_real));
  C2 = (nn_real *)malloc(M * N * sizeof(nn_real));

  cudaMalloc((void **)&dA, sizeof(nn_real) * M * K);
  cudaMalloc((void **)&dB, sizeof(nn_real) * K * N);
  cudaMalloc((void **)&dC1, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dC2, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dummy, sizeof(nn_real) * M * N);

  // C1 and C2 are same. We just have two copies to compare results
  createMATS(A, B, C1, C2, M, N, K);

  cudaMemcpy(dA, A, sizeof(nn_real) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(nn_real) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC1, C2, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC2, C2, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dummy, C2, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);

  /* Warm up GPU before we run. We run one extra CuBlas */
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle);

  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "CUBLAS initialization failed!" << std::endl;
    return;
  }

  stat = cublas_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M,
                     dB, K, &beta, dummy, M);

  /* Compute reference solution and time cuBLAS */
  using namespace std::chrono;
  high_resolution_clock::time_point ref_t1 = high_resolution_clock::now();

  for (int i = 0; i < NUM_ITERS; i++)
  {
    stat = cublas_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M,
                       dB, K, &beta, dC2, M);
  }

  check_launch("Reference GEMM");
  high_resolution_clock::time_point ref_t2 = high_resolution_clock::now();
  duration<double> ref_time_span =
      duration_cast<duration<double>>(ref_t2 - ref_t1);

  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "CUBLAS gemm error at " << __FILE__ << ":" << __LINE__
              << std::endl;
  }

  cudaMemcpy(C2, dC2, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  /* We are calling your GEMM function here */
  /* We will make one dummy call and check_launch here */
  int err;
  err = myGEMM(dA, dB, dummy, alpha, beta, M, N, K);
  check_launch("myGEMM dummy");

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  for (int i = 0; i < NUM_ITERS; i++)
  {
    err = myGEMM(dA, dB, dC1, alpha, beta, M, N, K);
  }

  check_launch("myGEMM");
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> my_time_span = duration_cast<duration<double>>(t2 - t1);

  /* This error code is for your own debugging, it does not catch
     illegal memory accesses or bad kernel launches */
  if (err != 0)
  {
    std::cout << "Error in my GEMM. Error code: " << err << std::endl;
  }

  cudaMemcpy(C1, dC1, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  int fail = compareGEMMResults(C1, C2, M, N);

  if (fail == 0)
  {
    std::cout << "Time for reference GEMM implementation: "
              << ref_time_span.count() << " seconds" << std::endl;
    std::cout << "Time for my GEMM implementation: " << my_time_span.count()
              << " seconds" << std::endl;
  }

  free(A);
  free(B);
  free(C1);
  free(C2);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC1);
  cudaFree(dC2);
  cudaFree(dummy);
}

void BenchmarkGEMM()
{
  std::cout << std::endl
            << "Entering GEMM Benchmarking mode! Stand by." << std::endl;

  /* First GEMM problem size */
  int M = 800 * SCALE, N = 1000 * SCALE, K = 784 * SCALE;

  std::cout << std::endl
            << "Starting GEMM 1: "
            << "M = " << M << "; N = " << N << "; K = " << K << std::endl;
  TestGEMM(M, N, K);
  std::cout << "Completed GEMM 1" << std::endl;

  /* Second GEMM problem size */
  M = 800 * SCALE, N = 100 * SCALE, K = 1000 * SCALE;
  std::cout << std::endl
            << "Starting GEMM 2: "
            << "M = " << M << "; N = " << N << "; K = " << K << std::endl;
  TestGEMM(M, N, K);
  std::cout << "Completed GEMM 2" << std::endl;

  /* Third GEMM problem size */
  M = 800 * SCALE, N = 10 * SCALE, K = 1000 * SCALE;
  std::cout << std::endl
            << "Starting GEMM 3: "
            << "M = " << M << "; N = " << N << "; K = " << K << std::endl;
  TestGEMM(M, N, K);
  std::cout << "Completed GEMM 3" << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
//                              SIGMOID TEST                                  //
////////////////////////////////////////////////////////////////////////////////


int compareSigmoidResults(nn_real *mySigmoid, nn_real *A, int M, int N)
{
  int i, j;
  int fail = 0;

  arma::Mat<nn_real> mysol = arma::Mat<nn_real>(mySigmoid, M, N, false);
  arma::Mat<nn_real> refA = arma::Mat<nn_real>(A, M, N, false);
  arma::Mat<nn_real> refSigmoid  = arma::Mat<nn_real>(M,N);
 
  refSigmoid = 1 / (1 + arma::exp(-refA));

  nn_real reldiff =
      arma::norm(mysol - refSigmoid, "inf") / arma::norm(refSigmoid, "inf");

  if (reldiff > TOL)
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "My Sigmoid output not matching with reference. Rel diff = "
              << reldiff << std::endl;
  }
  else
  {
    std::cout << "Sigmoid matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }

  return fail;
}



void TestSigmoid(int M, int N)
{
  nn_real *A;
  nn_real *Sigmoid;
  
  nn_real *dA;
  nn_real *dSigmoid;

  A = (nn_real *)malloc(M * N * sizeof(nn_real));
  Sigmoid = (nn_real *)malloc(M * N * sizeof(nn_real));

  cudaMalloc((void **)&dA, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dSigmoid, sizeof(nn_real) * M * N);

  createMAT(A, M, N);
  cudaMemcpy(dSigmoid, A, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);

  check_launch("Reference Sigmoid");

  int err;
  err = mySigmoid(dSigmoid, M, N);
  check_launch("mySigmoid dummy");

  cudaMemcpy(Sigmoid, dSigmoid, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  int fail = compareSigmoidResults(Sigmoid, A, M, N);

  free(A);
  free(Sigmoid);
  cudaFree(dA);
  cudaFree(dSigmoid);
}

void BenchmarkSigmoid()
{
  std::cout << std::endl
            << "Entering Sigmoid Benchmarking mode! Stand by." << std::endl;

  /* First Sigmoid problem size */
  int M = 800 * SCALE, N = 1000 * SCALE;

  std::cout << std::endl
            << "Starting sigmoid 1: "
            << "M = " << M << "; N = " << N << std::endl;
  TestSigmoid(M, N);
  std::cout << "Completed sigmoid 1" << std::endl;

  /* Second Sigmoid problem size */
  M = 800 * SCALE, N = 1;
  std::cout << std::endl
            << "Starting sigmoid 2: "
            << "M = " << M << "; N = " << N  << std::endl;
  TestSigmoid(M, N);
  std::cout << "Completed sigmoid 2" << std::endl;

  /* Third Sigmoid problem size */
  M = 5 * SCALE, N = 10 * SCALE;
  std::cout << std::endl
            << "Starting sigmoid 3: "
            << "M = " << M << "; N = " << N << std::endl;
  TestSigmoid(M, N);
  std::cout << "Completed sigmoid 3" << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
//                              SOFTMAX TEST                                  //
////////////////////////////////////////////////////////////////////////////////


// int compareSoftmaxResults(nn_real *mySoftmax, nn_real *A, int M, int N)
// {
//   int i, j;
//   int fail = 0;

//   arma::Mat<nn_real> mysol = arma::Mat<nn_real>(mySoftmax, M, N, false);
//   arma::Mat<nn_real> refA = arma::Mat<nn_real>(A, M, N, false);
//   arma::Mat<nn_real> refSoftmax  = arma::Mat<nn_real>(M,N);
 
//   arma::Mat<nn_real> exp_mat = arma::exp(refA);
//   arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
//   refSoftmax = exp_mat / repmat(sum_exp_mat, refA.n_rows, 1);

//   nn_real reldiff =
//       arma::norm(mysol - refSoftmax, "inf") / arma::norm(refSoftmax, "inf");

//   if (reldiff > TOL)
//   {
//     fail = 1;
//   }

//   // Print results
//   if (fail)
//   {
//     std::cout << "My Softmax output not matching with reference. Rel diff = "
//               << reldiff << std::endl;
//   }
//   else
//   {
//     std::cout << "Softmax matched with reference successfully! Rel diff = "
//               << reldiff << std::endl;
//   }

//   return fail;
// }



// void TestSoftmax(int M, int N)
// {
//   nn_real *A;
//   nn_real *Softmax;
  
//   nn_real *dA;
//   nn_real *dSoftmax;

//   A = (nn_real *)malloc(M * N * sizeof(nn_real));
//   Softmax = (nn_real *)malloc(M * N * sizeof(nn_real));

//   cudaMalloc((void **)&dA, sizeof(nn_real) * M * N);
//   cudaMalloc((void **)&dSoftmax, sizeof(nn_real) * M * N);

//   createMAT(A, M, N);
//   cudaMemcpy(dA, A, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);

//   check_launch("Reference Softmax");

//   int err;
//   err = mySoftmax(dA,dSoftmax, M, N);
//   check_launch("mySoftmax dummy");

//   cudaMemcpy(Softmax, dSoftmax, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

//   int fail = compareSoftmaxResults(Softmax, A, M, N);

//   free(A);
//   free(Softmax);
//   cudaFree(dA);
//   cudaFree(dSoftmax);
// }

// void BenchmarkSoftmax()
// {
//   std::cout << std::endl
//             << "Entering Softmax Benchmarking mode! Stand by." << std::endl;

//   /* First Softmax problem size */
//   int M = 800 * SCALE, N = 1000 * SCALE;

//   std::cout << std::endl
//             << "Starting Softmax 1: "
//             << "M = " << M << "; N = " << N << std::endl;
//   TestSoftmax(M, N);
//   std::cout << "Completed Softmax 1" << std::endl;

//   /* Second Softmax problem size */
//   M = 800 * SCALE, N = 1;
//   std::cout << std::endl
//             << "Starting Softmax 2: "
//             << "M = " << M << "; N = " << N  << std::endl;
//   TestSoftmax(M, N);
//   std::cout << "Completed Softmax 2" << std::endl;

//   /* Third Softmax problem size */
//   M = 5 * SCALE, N = 10 * SCALE;
//   std::cout << std::endl
//             << "Starting Softmax 3: "
//             << "M = " << M << "; N = " << N << std::endl;
//   TestSoftmax(M, N);
//   std::cout << "Completed Softmax 3" << std::endl;
// }

////////////////////////////////////////////////////////////////////////////////
//                              REPEAT  TEST                                  //
////////////////////////////////////////////////////////////////////////////////

int compareRepeatResults(nn_real *A, nn_real *B, nn_real *RepeatRow, nn_real *RepeatCol, int M, int N)
{
  int i, j;
  int fail = 0;

  arma::Mat<nn_real> myRepeatRow = arma::Mat<nn_real>(RepeatRow, M, N, false);
  arma::Mat<nn_real> myRepeatCol = arma::Mat<nn_real>(RepeatCol, M, N, false);
  arma::Mat<nn_real> solRepeatRow = arma::Mat<nn_real>(M, N);
  arma::Mat<nn_real> solRepeatCol = arma::Mat<nn_real>(M, N);
  arma::Mat<nn_real> refA = arma::Mat<nn_real>(A,1, N);
  arma::Mat<nn_real> refB = arma::Mat<nn_real>(B,M, 1);

  solRepeatRow = arma::repmat(refA, M, 1);
  solRepeatCol = arma::repmat(refB, 1, N);
  
  nn_real reldiffRow =
      arma::norm(myRepeatRow - solRepeatRow, "inf") / arma::norm(solRepeatRow, "inf");

  nn_real reldiffCol =
      arma::norm(myRepeatCol - solRepeatCol, "inf")/ arma::norm(solRepeatCol, "inf");


  if ((reldiffRow > TOL) || (reldiffCol > TOL))
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "MyRepeat output not matching with reference. Rel diff = "
              << reldiffRow << " " << reldiffCol << std::endl;
  }
  else
  {
    std::cout << "MyRepeat matched with reference successfully! Rel diff = "
              << reldiffRow << " " << reldiffCol << std::endl;
  }
  return fail;
}

void TestRepeat(int M, int N)
{
  nn_real *A;
  nn_real *B;
  nn_real *RepeatRow;
  nn_real *RepeatCol;

  nn_real *dA;
  nn_real *dB;
  nn_real *dRepeatRow;
  nn_real *dRepeatCol;

  RepeatRow = (nn_real *)malloc(M * N * sizeof(nn_real));
  RepeatCol = (nn_real *)malloc(M * N * sizeof(nn_real));
  A = (nn_real *)malloc(1 * N * sizeof(nn_real));
  B = (nn_real *)malloc(M * 1 * sizeof(nn_real));
  cudaMalloc((void **)&dRepeatRow, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dRepeatCol, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dA, sizeof(nn_real) * 1 * N);
  cudaMalloc((void **)&dB, sizeof(nn_real) * M * 1);

  createMAT(A, 1, N);
  createMAT(B, M, 1);

  cudaMemcpy(dA, A, sizeof(nn_real) * 1 * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(nn_real) * M * 1, cudaMemcpyHostToDevice);

  check_launch("Reference repeat");

  int err;
  err = myRepeatRow(dRepeatRow, dA, M, N);
  check_launch("row dummy");
  err = myRepeatCol(dRepeatCol, dB, M, N);
  check_launch("col dummy");

  cudaMemcpy(RepeatCol, dRepeatCol, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(RepeatRow, dRepeatRow, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  int fail = compareRepeatResults(A, B, RepeatRow, RepeatCol, M, N);

  free(RepeatRow);
  free(RepeatCol);
  free(A);
  free(B);
  cudaFree(dRepeatRow);
  cudaFree(dRepeatCol);
  cudaFree(dA);
  cudaFree(dB);
}


void BenchmarkRepeat()
{
  std::cout << std::endl
            << "Entering Repeat Benchmarking mode! Stand by." << std::endl;

  /* First Repeat problem size */
  int M = 800 * SCALE, N = 1000 * SCALE;

  std::cout << std::endl
            << "Starting Repeat 1: "
            << "M = " << M << "; N = " << N << std::endl;
  TestRepeat(M, N);
  std::cout << "Completed Repeat 1" << std::endl;

  /* Second Repeat problem size */
  M = 800 * SCALE, N = 1;
  std::cout << std::endl
            << "Starting Repeat 2: "
            << "M = " << M << "; N = " << N  << std::endl;
  TestRepeat(M, N);
  std::cout << "Completed Repeat 2" << std::endl;

  /* Third Repeat problem size */
  M = 5 * SCALE, N = 10 * SCALE;
  std::cout << std::endl
            << "Starting Repeat 3: "
            << "M = " << M << "; N = " << N << std::endl;
 TestRepeat(M, N);
  std::cout << "Completed Repeat 3" << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
//                              TRANSPOSE  TEST                               //
////////////////////////////////////////////////////////////////////////////////

int compareTransposeResults(nn_real *C, nn_real *A,int M, int N)
{
  int i, j;
  int fail = 0;

  arma::Mat<nn_real> mysol = arma::Mat<nn_real>(C, N, M, false);
  arma::Mat<nn_real> refA = arma::Mat<nn_real>(A, M, N, false);
  arma::Mat<nn_real> refsol = arma::Mat<nn_real>(N,M);

  refsol = refA.t();
  nn_real reldiff =
      arma::norm(mysol - refsol, "inf") / arma::norm(refsol, "inf");

  if (reldiff > TOL)
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "My Transpose output not matching with reference. Rel diff = "
              << reldiff << std::endl;
  }
  else
  {
    std::cout << "Transpose matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }
  return fail;
}

void TestTranspose(int M, int N)
{
  nn_real *A = nullptr;
  nn_real *B = nullptr;

  nn_real *dA = nullptr;
  nn_real *dB = nullptr;


  A = (nn_real *)malloc(M * N * sizeof(nn_real));
  B = (nn_real *)malloc(M * N * sizeof(nn_real));

  cudaMalloc((void **)&dA, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dB, sizeof(nn_real) * M * N);

  createMAT(A, M, N);

  cudaMemcpy(dA, A, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);

  check_launch("Reference transpose");

  int err;
  myTranspose(dA, dB, M, N);
  check_launch("myTranspose dummy");

  cudaMemcpy(B, dB, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  int fail = compareTransposeResults(B, A, M, N);

  free(A);
  free(B);
  cudaFree(dA);
  cudaFree(dB);
}



void BenchmarkTranspose()
{
  std::cout << std::endl
            << "Entering Transpose Benchmarking mode! Stand by." << std::endl;


  /* First Transpose problem size */
  int M = 800 * SCALE, N = 1000 * SCALE;

  std::cout << std::endl
            << "Starting Transpose 1: "
            << "M = " << M << "; N = " << N << std::endl;
  TestTranspose(M, N);
  std::cout << "Completed Transpose 1" << std::endl;

  /* Second Transpose problem size */
  M = 800 * SCALE, N = 1;
  std::cout << std::endl
            << "Starting Transpose 2: "
            << "M = " << M << "; N = " << N  << std::endl;
  TestTranspose(M, N);
  std::cout << "Completed Transpose 2" << std::endl;

  /* Third Transpose problem size */
  M = 5 * SCALE, N = 10 * SCALE;
  std::cout << std::endl
            << "Starting Transpose 3: "
            << "M = " << M << "; N = " << N << std::endl;
 TestTranspose(M, N);
  std::cout << "Completed Transpose 3" << std::endl;
}



////////////////////////////////////////////////////////////////////////////////
//                              DIFFERENCE TEST                               //
////////////////////////////////////////////////////////////////////////////////

int compareDifferenceResults(nn_real *A, nn_real *B, nn_real *C, int M, int N)
{
  int i, j;
  int fail = 0;

  arma::Mat<nn_real> mysol = arma::Mat<nn_real>(C, M, N, false);
  arma::Mat<nn_real> refA = arma::Mat<nn_real>(A, M, N, false);
  arma::Mat<nn_real> refB = arma::Mat<nn_real>(B, M, N, false);
  arma::Mat<nn_real> refsol = arma::Mat<nn_real>(M,N);

  refsol = refA - refB;
  nn_real reldiff =
      arma::norm(mysol - refsol, "inf") / arma::norm(refsol, "inf");

  if (reldiff > TOL)
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "My Difference output not matching with reference. Rel diff = "
              << reldiff << std::endl;
  }
  else
  {
    std::cout << "Difference matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }
  return fail;
}

void TestDifference(int M, int N)
{
  nn_real *A = nullptr;
  nn_real *B = nullptr;
  nn_real *C = nullptr;

  nn_real *dA = nullptr;
  nn_real *dB = nullptr;
  nn_real *dC = nullptr;


  A = (nn_real *)malloc(M * N * sizeof(nn_real));
  B = (nn_real *)malloc(M * N * sizeof(nn_real));
  C = (nn_real *)malloc(M * N * sizeof(nn_real));

  cudaMalloc((void **)&dA, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dB, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dC, sizeof(nn_real) * M * N);

  createMAT(A, M, N);
  createMAT2(B, M, N);

  cudaMemcpy(dA, A, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);


  check_launch("Reference Difference");

  int err;
  myDifference(dA, dB, dC, M, N);
  check_launch("myDifference dummy");

  cudaMemcpy(C, dC, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  int fail = compareDifferenceResults(A, B, C, M, N);

  free(A);
  free(B);
  free(C);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}



void BenchmarkDifference()
{
  std::cout << std::endl
            << "Entering Difference Benchmarking mode! Stand by." << std::endl;

  /* First Difference problem size */
  int M = 800 * SCALE, N = 1000 * SCALE;

  std::cout << std::endl
            << "Starting Difference 1: "
            << "M = " << M << "; N = " << N << std::endl;
  TestDifference(M, N);
  std::cout << "Completed Difference 1" << std::endl;

  /* Second Difference problem size */
  M = 800 * SCALE, N = 1;
  std::cout << std::endl
            << "Starting Difference 2: "
            << "M = " << M << "; N = " << N  << std::endl;
  TestDifference(M, N);
  std::cout << "Completed Difference 2" << std::endl;

  /* Third Difference problem size */
  M = 5 * SCALE, N = 10 * SCALE;
  std::cout << std::endl
            << "Starting Difference 3: "
            << "M = " << M << "; N = " << N << std::endl;
 TestDifference(M, N);
  std::cout << "Completed Difference 3" << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
//                              HADAMARD PRODUCT TEST                         //
////////////////////////////////////////////////////////////////////////////////

int compareHadamardResults(nn_real *A, nn_real *B, nn_real *sol,  int M, int N)
{
  int i, j;
  int fail = 0;

  arma::Mat<nn_real> mysol = arma::Mat<nn_real>(sol, M, N, false);
  arma::Mat<nn_real> refA = arma::Mat<nn_real>(A, M, N, false);
  arma::Mat<nn_real> refB = arma::Mat<nn_real>(B, M, N, false);
  arma::Mat<nn_real> refsol = arma::Mat<nn_real>(M,N);

  refsol = refA % refB;
  nn_real reldiff =
      arma::norm(mysol - refsol, "inf") / arma::norm(refsol, "inf");

  if (reldiff > TOL)
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "My Hadamard output not matching with reference. Rel diff = "
              << reldiff << std::endl;
  }
  else
  {
    std::cout << "Hadamard matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }
  return fail;
}

void TestHadamard(int M, int N)
{
  nn_real *A = nullptr;
  nn_real *B = nullptr;
  nn_real *A_cpy = nullptr;

  nn_real *dA = nullptr;
  nn_real *dB = nullptr;


  A = (nn_real *)malloc(M * N * sizeof(nn_real));
  A_cpy = (nn_real *)malloc(M * N * sizeof(nn_real));
  B = (nn_real *)malloc(M * N * sizeof(nn_real));

  cudaMalloc((void **)&dA, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dB, sizeof(nn_real) * M * N);

  createMAT(A, M, N);
  createMAT(A_cpy, M, N);
  createMAT2(B, M, N);

  cudaMemcpy(dA, A, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);


  check_launch("Reference Hadamard");

  int err;
  myHadamardProduct(dA, dB, M, N);
  check_launch("myHadamard dummy");

  cudaMemcpy(A, dA, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  int fail = compareHadamardResults(A_cpy, B, A, M, N);

  free(A);
  free(B);
  cudaFree(dA);
  cudaFree(dB);
}



void BenchmarkHadamard()
{
  std::cout << std::endl
            << "Entering Hadamard Benchmarking mode! Stand by." << std::endl;

  /* First Hadamard problem size */
  int M = 800 * SCALE, N = 1000 * SCALE;

  std::cout << std::endl
            << "Starting Hadamard 1: "
            << "M = " << M << "; N = " << N << std::endl;
  TestHadamard(M, N);
  std::cout << "Completed Hadamard 1" << std::endl;

  /* Second Hadamard problem size */
  M = 800 * SCALE, N = 1;
  std::cout << std::endl
            << "Starting Hadamard 2: "
            << "M = " << M << "; N = " << N  << std::endl;
  TestHadamard(M, N);
  std::cout << "Completed Hadamard 2" << std::endl;

  /* Third Hadamard problem size */
  M = 5 * SCALE, N = 10 * SCALE;
  std::cout << std::endl
            << "Starting Hadamard 3: "
            << "M = " << M << "; N = " << N << std::endl;
 TestHadamard(M, N);
  std::cout << "Completed Hadamard 3" << std::endl;
}



////////////////////////////////////////////////////////////////////////////////
//                              SUM ROW TEST                                  //
////////////////////////////////////////////////////////////////////////////////


int compareSumRowResults(nn_real *A, nn_real *B, int M, int N)
{
  int i, j;
  int fail = 0;

  arma::Mat<nn_real> mysol = arma::Mat<nn_real>(B, M, 1, false);
  arma::Mat<nn_real> refA = arma::Mat<nn_real>(A, M, N, false);
  arma::Mat<nn_real> refsol = arma::Mat<nn_real>(M,1);

  refsol = arma::sum(refA, 1);
  nn_real reldiff =
      arma::norm(mysol - refsol, "inf") / arma::norm(refsol, "inf");

  if (reldiff > TOL)
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "My SumRow output not matching with reference. Rel diff = "
              << reldiff << std::endl;
  }
  else
  {
    std::cout << "SumRow matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }
  return fail;
}


// int mySumRow(nn_real *A, nn_real *B, int M, int N){
    // sum each row of A, dst = B, dimensions MxN

void TestSumRow(int M, int N)
{
  nn_real *A = nullptr;
  nn_real *B = nullptr;

  nn_real *dA = nullptr;
  nn_real *dB = nullptr;


  A = (nn_real *)malloc(M * N * sizeof(nn_real));
  B = (nn_real *)malloc(M * N * sizeof(nn_real));

  cudaMalloc((void **)&dA, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dB, sizeof(nn_real) * M * 1);

  createMAT(A, M, N);

  cudaMemcpy(dA, A, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(nn_real) * M * 1, cudaMemcpyHostToDevice);

  check_launch("Reference SumRow");

  int err;
  mySumRow(dA, dB, M, N);
  check_launch("mySumRow dummy");

  cudaMemcpy(B, dB, sizeof(nn_real) * M * 1, cudaMemcpyDeviceToHost);

  int fail = compareSumRowResults(A, B, M, N);

  free(A);
  free(B);
  cudaFree(dA);
  cudaFree(dB);
}



void BenchmarkSumRow()
{
  std::cout << std::endl
            << "Entering SumRow Benchmarking mode! Stand by." << std::endl;

  /* First SumRow problem size */
  int M = 800 * SCALE, N = 1000 * SCALE;

  std::cout << std::endl
            << "Starting SumRow 1: "
            << "M = " << M << "; N = " << N << std::endl;
  TestSumRow(M, N);
  std::cout << "Completed SumRow 1" << std::endl;

  /* Second SumRow problem size */
  M = 800 * SCALE, N = 1;
  std::cout << std::endl
            << "Starting SumRow 2: "
            << "M = " << M << "; N = " << N  << std::endl;
  TestSumRow(M, N);
  std::cout << "Completed SumRow 2" << std::endl;

  /* Third SumRow problem size */
  M = 5 * SCALE, N = 10 * SCALE;
  std::cout << std::endl
            << "Starting SumRow 3: "
            << "M = " << M << "; N = " << N << std::endl;
 TestSumRow(M, N);
  std::cout << "Completed SumRow 3" << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
//                              ONE MINUS TEST                                //
////////////////////////////////////////////////////////////////////////////////

// myOneMinus(nn.dsigdx, nn.d_a1, nn.d_H1, nn.d_N);

int compareOneMinusResults(nn_real *A, nn_real *B, int M, int N)
{
  int i, j;
  int fail = 0;

  arma::Mat<nn_real> mysol = arma::Mat<nn_real>(B, M, N, false);
  arma::Mat<nn_real> refA = arma::Mat<nn_real>(A, M, N, false);
  arma::Mat<nn_real> refsol = arma::Mat<nn_real>(M,N);


  refsol = 1 - refA;
  nn_real reldiff =
      arma::norm(mysol - refsol, "inf") / arma::norm(refsol, "inf");

  if (reldiff > TOL)
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "My OneMinus output not matching with reference. Rel diff = "
              << reldiff << std::endl;
  }
  else
  {
    std::cout << "OneMinus matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }
  return fail;
}


// int myOneMinus(nn_real *A, nn_real *B, int M, int N){
    // sum each row of A, dst = B, dimensions MxN

void TestOneMinus(int M, int N)
{
  nn_real *A = nullptr;
  nn_real *B = nullptr;

  nn_real *dA = nullptr;
  nn_real *dB = nullptr;


  A = (nn_real *)malloc(M * N * sizeof(nn_real));
  B = (nn_real *)malloc(M * N * sizeof(nn_real));

  cudaMalloc((void **)&dA, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dB, sizeof(nn_real) * M * N);

  createMAT(A, M, N);

  cudaMemcpy(dA, A, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);

  check_launch("Reference OneMinus");

  int err;
  myOneMinus(dA, dB, M, N);
  check_launch("myOneMinus dummy");

  cudaMemcpy(B, dB, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  int fail = compareOneMinusResults(A, B, M, N);

  free(A);
  free(B);
  cudaFree(dA);
  cudaFree(dB);
}



void BenchmarkOneMinus()
{
  std::cout << std::endl
            << "Entering OneMinus Benchmarking mode! Stand by." << std::endl;

 
  /* First OneMinus problem size */
  int M = 800 * SCALE, N = 1000 * SCALE;

  std::cout << std::endl
            << "Starting OneMinus 1: "
            << "M = " << M << "; N = " << N << std::endl;
  TestOneMinus(M, N);
  std::cout << "Completed OneMinus 1" << std::endl;

  /* Second OneMinus problem size */
  M = 800 * SCALE, N = 1;
  std::cout << std::endl
            << "Starting OneMinus 2: "
            << "M = " << M << "; N = " << N  << std::endl;
  TestOneMinus(M, N);
  std::cout << "Completed OneMinus 2" << std::endl;

  /* Third OneMinus problem size */
  M = 5 * SCALE, N = 10 * SCALE;
  std::cout << std::endl
            << "Starting OneMinus 3: "
            << "M = " << M << "; N = " << N << std::endl;
 TestOneMinus(M, N);
  std::cout << "Completed OneMinus 3" << std::endl;
}