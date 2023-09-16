#include "utils/neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>
#include <iomanip>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "mpi.h"

#define TOL 1e-14 // Tolerance for tests
#define DEBUG 0



#define MPI_SAFE_CALL(call)                                                  \
  do                                                                         \
  {                                                                          \
    int err = call;                                                          \
    if (err != MPI_SUCCESS)                                                  \
    {                                                                        \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

int get_num_batches(int N, int Batch_size)
{
  return (N + Batch_size - 1) / Batch_size;
}

int get_batch_size(int N, int Batch_size, int batch)
{
  int num_batches = get_num_batches(N, Batch_size);
  return (batch == num_batches - 1) ? N - Batch_size * batch : Batch_size;
}

int get_mini_batch_size(int Batch_size, int num_procs, int rank)
{
  int mini_Batch_size = Batch_size / num_procs;
  return rank < Batch_size % num_procs ? mini_Batch_size + 1 : mini_Batch_size;
}

nn_real norms(NeuralNetwork &nn)
{
  nn_real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i)
  {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

/* CPU implementation.
 * Follow this code to build your GPU code.
 */

// Sigmoid activation
void sigmoid(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  ASSERT_MAT_SAME_SIZE(mat, mat2);
  mat2 = 1 / (1 + arma::exp(-mat));
}

// Softmax activation
void softmax(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  arma::Mat<nn_real> exp_mat = arma::exp(mat);
  arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
  mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
}

// feedforward pass
void feedforward(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
                 struct cache &cache)
{
  cache.z.resize(2);
  cache.a.resize(2);

  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<nn_real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<nn_real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<nn_real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<nn_real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork &nn, const arma::Mat<nn_real> &y, nn_real reg,
              const struct cache &bpcache, struct grads &bpgrads)
{
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<nn_real> da1 = nn.W[1].t() * diff;

  arma::Mat<nn_real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
nn_real loss(NeuralNetwork &nn, const arma::Mat<nn_real> &yc,
             const arma::Mat<nn_real> &y, nn_real reg)
{
  int N = yc.n_cols;
  nn_real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  nn_real data_loss = ce_sum / N;
  nn_real reg_loss = 0.5 * reg * norms(nn);
  nn_real loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             arma::Row<nn_real> &label)
{
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i)
  {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
           const arma::Mat<nn_real> &y, nn_real learning_rate, nn_real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug)
{
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  assert(X.n_cols == y.n_cols);

  int num_batches = get_num_batches(N, batch_size);

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    int batch_start = 0;
    for (int batch = 0; batch < num_batches; ++batch)
    {
      int last_col = batch_start + get_batch_size(N, batch_size, batch);
      assert(last_col <= X.n_cols);
      assert(last_col <= y.n_cols);
      assert(last_col > batch_start);
      if (batch == num_batches - 1)
      {
        assert(last_col == X.n_cols);
      }
      arma::Mat<nn_real> X_batch = X.cols(batch_start, last_col - 1);
      arma::Mat<nn_real> y_batch = y.cols(batch_start, last_col - 1);

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);

      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0)
      {
        if (grad_check)
        {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
        }

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i)
      {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i)
      {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to the "cpu_save_dir" folder.
         In the later runs (with same parameters), you can use just the debug
         flag to output diff b/w CPU and GPU without running the CPU version
         version. */
      if (print_every <= 0)
      {
        print_flag = batch == 0;
      }
      else
      {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag)
      {
        save_cpu_data(nn, iter);
      }

      batch_start = last_col;
      iter++;
    }
  }
}

class parallel_NeuralNetwork {
  public:
    nn_real* d_W1 = nullptr;
    nn_real* d_W2 = nullptr;
    nn_real* d_b1 = nullptr;
    nn_real* d_b2 = nullptr;
    nn_real* d_a1 = nullptr;

    nn_real* d_gradW2 = nullptr;
    nn_real* d_grada1 = nullptr;
    nn_real* d_grada2 = nullptr;
    nn_real* d_gradb1 = nullptr;
    nn_real* d_gradb2 = nullptr;
    nn_real* d_gradz2 = nullptr;
    nn_real* d_gradW1 = nullptr;

    nn_real* d_yc = nullptr;
    nn_real* d_gradz1 = nullptr;



    int d_H0; int d_H1; int d_H2; int d_N;
    const int layers = 2;

    parallel_NeuralNetwork(int H0, int H1, int H2, int N); 
    ~parallel_NeuralNetwork();

    void toGPU(NeuralNetwork& nn);
    void fromGPU(nn_real *h_gradW1, nn_real *h_gradW2, nn_real *h_gradb1, nn_real *h_gradb2);
};


parallel_NeuralNetwork::parallel_NeuralNetwork(int H0, int H1, int H2, int N)
{
  cudaError_t err;

  //// Neural network components
  err =cudaMalloc(&d_W1, sizeof(nn_real) * H0 * H1);
  if (err != cudaSuccess)
  {
    std::cerr << "malloc d_W1" << std::endl;
    exit(1);
  }

  err = cudaMalloc(&d_W2, sizeof(nn_real) * H1 * H2);
  if (err != cudaSuccess)
  {
    std::cerr << "malloc d_W2" << std::endl;
    exit(1);
  }

  err = cudaMalloc(&d_b1, sizeof(nn_real) * H1 * 1);
  if (err != cudaSuccess)
  {
    std::cerr << "malloc d_b1" << std::endl;
    exit(1);
  }

  err = cudaMalloc(&d_b2, sizeof(nn_real) * H2 * 1);

  if (err != cudaSuccess)
  {
    std::cerr << "malloc d_b2" << std::endl;
    exit(1);
  }

  err = cudaMalloc(&d_a1, sizeof(nn_real) * H1 * N);
  if (err != cudaSuccess)
  {
    std::cerr << "malloc d_a1" << std::endl;
    exit(1);
  }


    err = cudaMalloc(&d_yc, sizeof(nn_real) * H2 * N);

  if (err != cudaSuccess)
  {
    std::cerr << "malloc d_yc" << std::endl;
    exit(1);
  }


  //// Gradients
  err = cudaMalloc(&d_grada1, sizeof(nn_real) * H1 * N);
  if (err != cudaSuccess)
  {
    std::cerr << "malloc d_grada1" << std::endl;
    exit(1);
  }

  err = cudaMalloc(&d_gradb1, sizeof(nn_real) * H1 * 1);

  if (err != cudaSuccess)
  {
    std::cerr << "malloc d_gradb1" << std::endl;
    exit(1);
  }

  err = cudaMalloc(&d_gradb2, sizeof(nn_real) * H2 * 1);

  if (err != cudaSuccess)
  {
    std::cerr << "malloc d_gradb2" << std::endl;
    exit(1);
  }

  err = cudaMalloc(&d_gradW1, sizeof(nn_real) * H0 * H1);

  if (err != cudaSuccess)
  {
    std::cerr << "malloc d_gradW1" << std::endl;
    exit(1);
  }

  err = cudaMalloc(&d_gradW2, sizeof(nn_real) * H1 * H2);

  if (err != cudaSuccess)
  {
      std::cerr << "malloc d_gradW2" << std::endl;
      exit(1);
  }

  err = cudaMalloc(&d_gradz1, sizeof(nn_real) * H1 * N);

  if (err != cudaSuccess)
  {
      std::cerr << "malloc d_gradz1" << std::endl;
      exit(1);
  }

  d_H0 = H0;
  d_H1 = H1;
  d_H2 = H2;
  d_N  = N;

}

// cuda free memory
parallel_NeuralNetwork::~parallel_NeuralNetwork() {
  cudaFree(d_W1);
  cudaFree(d_W2);
  cudaFree(d_b1);
  cudaFree(d_b2);

  cudaFree(d_a1);
  cudaFree(d_yc);

  cudaFree(d_gradW1);
  cudaFree(d_gradW2);
  cudaFree(d_gradb1);
  cudaFree(d_gradb2);
  cudaFree(d_grada1);
}

// copy from host -> device
void parallel_NeuralNetwork::toGPU(NeuralNetwork& nn) { 
 
  cudaError_t err;	  
  err = cudaMemcpy(d_W1, nn.W[0].memptr(), d_H0 * d_H1 * sizeof(nn_real), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    std::cerr << "Could not copy W1 to GPU" << std::endl;
    exit(1);
  }

  err = cudaMemcpy(d_W2, nn.W[1].memptr(), d_H1 * d_H2 * sizeof(nn_real), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    std::cerr << "Could not copy W2 to GPU" << std::endl;
    exit(1);
  }

  err = cudaMemcpy(d_b1, nn.b[0].memptr(), d_H1 * 1 * sizeof(nn_real), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    std::cerr << "Could not copy b1 to GPU" << std::endl;
    exit(1);
  }

  err = cudaMemcpy(d_b2, nn.b[1].memptr(), d_H2 * 1 * sizeof(nn_real), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    std::cerr << "Could not copy b2 to GPU" << std::endl;
    exit(1);
  }

  err = cudaMemcpy(d_gradW1, nn.W[0].memptr(), d_H0 * d_H1 * sizeof(nn_real), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    std::cerr << "Could not copy gradW1 to GPU" << std::endl;
    exit(1);
  }


  err = cudaMemcpy(d_gradW2, nn.W[1].memptr(), d_H1 * d_H2 * sizeof(nn_real), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    std::cerr << "Could not copy gradW2 to GPU" << std::endl;
    exit(1);
  }
}

// copy from device -> host
void parallel_NeuralNetwork::fromGPU(nn_real *h_gradW1, nn_real *h_gradW2, nn_real *h_gradb1, nn_real *h_gradb2) {

cudaError_t err;
err = cudaMemcpy(h_gradW1, d_gradW1, d_H1 * d_H0 * sizeof(nn_real), cudaMemcpyDeviceToHost);
if (err != cudaSuccess)
{
  std::cerr << "Could not copy gradW1 from GPU" << std::endl;
  exit(1);
}

err = cudaMemcpy(h_gradW2, d_gradW2, d_H2 * d_H1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
if (err != cudaSuccess)
{
  std::cerr << "Could not copy gradW2 from GPU" << std::endl;
  exit(1);
}

err = cudaMemcpy(h_gradb1, d_gradb1, d_H1 * 1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
if (err != cudaSuccess)
{
  std::cerr << "Could not copy gradb1 from GPU" << std::endl;
  exit(1);
}

err = cudaMemcpy(h_gradb2, d_gradb2, d_H2 * 1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
if (err != cudaSuccess)
{
  std::cerr << "Could not copy gradb2 from GPU" << std::endl;
  exit(1);
}
  
}



void parallel_backprop(parallel_NeuralNetwork& nn, nn_real *X, nn_real *y, nn_real reg, 
int minibatch_size, int batch_size)
{
  myScalarDifference(nn.d_yc, y, nn.d_H2, minibatch_size, 1.0 / batch_size); 

  myGEMMSumRowTranspose(nn.d_yc, nn.d_a1, nn.d_gradW2,1.0, reg, nn.d_H2, nn.d_H1, minibatch_size, nn.d_gradb2);

  myGEMMTranspose(nn.d_W2,nn.d_yc,nn.d_grada1,1.0,0.0,nn.d_H1,minibatch_size, nn.d_H2);

  myOneMinusHadamard(nn.d_gradz1, nn.d_a1, nn.d_grada1, nn.d_H1, minibatch_size); 

  myGEMMSumRowTranspose(nn.d_gradz1,X,nn.d_gradW1,1.0,reg, nn.d_H1, nn.d_H0, minibatch_size, nn.d_gradb1);

}


void parallel_feedforward(parallel_NeuralNetwork &nn, nn_real* X, int N)
{

  // myRepeatCol(nn.d_a1,nn.d_b1,nn.d_H1,N); // dst, src
  // myGEMMSigmoid(nn.d_W1,X,nn.d_a1,1.0,1.0,nn.d_H1, N, nn.d_H0);

  myGEMMSigmoidRepmat(nn.d_W1,X,nn.d_b1,1.0,1.0,nn.d_H1, N, nn.d_H0, nn.d_a1);

  // myRepeatCol(nn.d_yc,nn.d_b2,nn.d_H2, N);
  // myGEMM(nn.d_W2,nn.d_a1,nn.d_yc,1.0,1.0,nn.d_H2,N, nn.d_H1); 

  myGEMMRepmat(nn.d_W2,nn.d_a1,nn.d_b2,1.0,1.0,nn.d_H2,N, nn.d_H1, nn.d_yc); 

  mySoftmax(nn.d_yc, nn.d_H2, N); 
}

/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
                    const arma::Mat<nn_real> &y, nn_real learning_rate,
                    std::ofstream &error_file, nn_real reg, const int epochs,
                    const int batch_size, int print_every, int debug)
{
  assert(learning_rate > 0);
  assert(reg >= 0);
  assert(epochs >= 0);
  assert(batch_size > 0);

  int rank, num_procs;
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (rank == 0)
  {
    assert(X.n_cols > 0);
    assert(X.n_rows == IMAGE_SIZE);
    assert(y.n_cols == X.n_cols);
    assert(y.n_rows == NUM_CLASSES);
    assert(nn.H[0] == IMAGE_SIZE);
    assert(nn.H[2] == NUM_CLASSES);
  }


  int N = (rank == 0) ? X.n_cols : 0;

  // if (num_procs > 1)
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

  assert(N > 0);

  int print_flag = 0;

  /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way using memptr().
     Or you can allocate your own array memory space and store the elements in a
     row major way. Remember to update the Armadillo matrices in NeuralNetwork
     &nn of rank 0 before returning from the function. */

  /* allocate memory before the iterations */
  // Data sets
  const int num_batches = get_num_batches(N, batch_size);
  int mini_batch_size_alloc;
  {
    const int max_batch_size = batch_size;
    mini_batch_size_alloc = max_batch_size / num_procs + 1;
  }

  // TODO

  int H0 = nn.H[0];
  int H1 = nn.H[1];
  int H2 = nn.H[2];

  int X_nrows = X.n_rows;
  int X_ncols = X.n_cols;
  int y_nrows = y.n_rows;

  // if (num_procs > 1){
    MPI_SAFE_CALL(MPI_Bcast(&X_nrows, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Bcast(&y_nrows, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Bcast(&X_ncols, 1, MPI_INT, 0, MPI_COMM_WORLD));
  // }

  // int* Xsendcounts;
  // int* Ysendcounts;
  // int* Xdispls;
  // int* Ydispls;


  int minibatch_size = mini_batch_size_alloc - 1;

  // vector of pointers containing each minibatch
  std::vector<nn_real*> dX_minibatch(num_batches,nullptr);
  std::vector<nn_real*> dy_minibatch(num_batches,nullptr);
  
  nn_real *h_gradW1;
  nn_real *h_gradW2;
  nn_real *h_gradb1;
  nn_real *h_gradb2;
  nn_real *d_Xbatch;
  nn_real *d_Ybatch;

  nn_real *h_Xbatch;
  nn_real *h_Ybatch;
  nn_real *h_yc;

  int Batch_size = batch_size;

  h_gradW1 = (nn_real *)malloc(H0 * H1 * sizeof(nn_real));
  h_gradW2 = (nn_real *)malloc(H1 * H2 * sizeof(nn_real));
  h_gradb1 = (nn_real *)malloc(H1 * 1 * sizeof(nn_real));
  h_gradb2 = (nn_real *)malloc(H2 * 1 * sizeof(nn_real));
  h_yc = (nn_real *)malloc(H2 * Batch_size * sizeof(nn_real));


  int batch_start = 0;
  int data_batch_size, data_minibatch_size, max_minibatch_size;


  for (int batch = 0; batch < num_batches; ++batch){
    // figure out batch/minibatch size
    int last_col = batch_start + get_batch_size(N, batch_size, batch);
    data_batch_size = last_col  - batch_start;
    data_minibatch_size = data_batch_size / num_procs;

    // declare size of the matrices
    arma::Mat<nn_real> X_batch(X_nrows, data_batch_size);
    arma::Mat<nn_real> y_batch(y_nrows, data_batch_size);

    // arma::Row<nn_real> X_batch(X_nrows * data_batch_size);
    // arma::Row<nn_real> y_batch(y_nrows * data_batch_size);


    // arma::Mat<nn_real> X_minibatch(X_nrows, data_minibatch_size + 1);
    // arma::Mat<nn_real> y_minibatch(y_nrows, data_minibatch_size + 1);

    arma::Mat<nn_real> X_minibatch(X_nrows, data_minibatch_size);
    arma::Mat<nn_real> y_minibatch(y_nrows, data_minibatch_size);

    // arma::Row<nn_real> X_minibatch(X_nrows *data_minibatch_size );
    // arma::Row<nn_real> y_minibatch(y_nrows *data_minibatch_size );



    if( rank == 0){
      X_batch = X.cols(batch_start, last_col - 1);
      y_batch = y.cols(batch_start, last_col - 1);
    }

    // SENDCOUNTS(I) is the number of items of type SENDTYPE to send from process ROOT to process I. 
    // DISPLS(I) is the displacement from SENDBUF to the beginning of the I-th message, in units of SENDTYPE. 

    // std::cout << "Ẍ_nrows" << X_nrows << std::endl;
    // if (rank == 0) {
    //     int Xsum = 0;
    //     Xsendcounts = (int*)malloc(sizeof(int)*num_procs); 
    //     Xdispls = (int*)malloc(sizeof(int)*num_procs); 
    //     for (int i = 0; i < num_procs; ++i) {
    //       Xsendcounts[i] = X_nrows*get_mini_batch_size(get_batch_size(N, batch_size, batch), num_procs, i);
    //       Xdispls[i] = Xsum;
    //       Xsum += Xsendcounts[i];
    //     }

    //     int Ysum = 0;
    //     Ysendcounts = (int*)malloc(sizeof(int)*num_procs); 
    //     Ydispls = (int*)malloc(sizeof(int)*num_procs); 
    //     for (int i = 0; i < num_procs; ++i) {
    //       Ysendcounts[i] = nn.H[2]*get_mini_batch_size(get_batch_size(N, batch_size, batch), num_procs, i);
    //       Ydispls[i] = Ysum;
    //       Ysum += Ysendcounts[i];
    //     }
    //   }

    // // scatter the data between the processors with X_minibatch pointer
    // MPI_SAFE_CALL(MPI_Scatter(X_batch.memptr(),X_nrows*data_minibatch_size, MPI_FP, X_minibatch.memptr(),X_nrows*data_minibatch_size,MPI_FP, 0, MPI_COMM_WORLD));
    // MPI_SAFE_CALL(MPI_Scatter(y_batch.memptr(),y_nrows*data_minibatch_size, MPI_FP,y_minibatch.memptr(),y_nrows*data_minibatch_size, MPI_FP, 0, MPI_COMM_WORLD)); 

    // // std::cout << "data minibatch"<< data_minibatch_size << std::endl;

    // // MPI_SAFE_CALL(MPI_Scatterv(X_batch.memptr(), Xsendcounts, Xdispls, MPI_FP, X_minibatch.memptr(), X_nrows*(data_minibatch_size+1), MPI_FP, 0, MPI_COMM_WORLD));
    // // MPI_SAFE_CALL(MPI_Scatterv(y_batch.memptr(), Ysendcounts, Ydispls, MPI_FP, y_minibatch.memptr(), nn.H[2]*(data_minibatch_size+1), MPI_FP, 0, MPI_COMM_WORLD));

    // // allocate the size of the minibatch on the device at the dX_minibatch[batch] location
    // cudaMalloc((void **)&dX_minibatch[batch], sizeof(nn_real) * H0 * data_minibatch_size);
    // cudaMalloc((void **)&dy_minibatch[batch], sizeof(nn_real) * H2 * data_minibatch_size);

    // // copy the data of the minibatch to the device
    // cudaMemcpy(dX_minibatch[batch], X_minibatch.memptr(), sizeof(nn_real) * H0 * data_minibatch_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(dy_minibatch[batch], y_minibatch.memptr(), sizeof(nn_real) * H2 * data_minibatch_size, cudaMemcpyHostToDevice);

    // // std::cout << std::endl << std::endl << std::endl;


        // scatter the data between the processors with X_minibatch pointer
    if (num_procs > 1){
    MPI_SAFE_CALL(MPI_Scatter(X_batch.memptr(),X_nrows*data_minibatch_size, MPI_FP, X_minibatch.memptr(),X_nrows*data_minibatch_size,MPI_FP, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Scatter(y_batch.memptr(),y_nrows*data_minibatch_size, MPI_FP,y_minibatch.memptr(),y_nrows*data_minibatch_size, MPI_FP, 0, MPI_COMM_WORLD)); 
    }
    else{
      X_minibatch = X_batch;
      y_minibatch = y_batch;
    }

    // allocate the size of the minibatch on the device at the dX_minibatch[batch] location
    cudaMalloc((void **)&dX_minibatch[batch], sizeof(nn_real) * H0 * data_minibatch_size);
    cudaMalloc((void **)&dy_minibatch[batch], sizeof(nn_real) * H2 * data_minibatch_size);

    // copy the data of the minibatch to the device
    cudaMemcpy(dX_minibatch[batch], X_minibatch.memptr(), sizeof(nn_real) * H0 * data_minibatch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dy_minibatch[batch], y_minibatch.memptr(), sizeof(nn_real) * H2 * data_minibatch_size, cudaMemcpyHostToDevice);


    batch_start = last_col;

  }

  parallel_NeuralNetwork dnn(H0,H1,H2, minibatch_size);
  dnn.toGPU(nn); 




  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;
  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    int batch_start = 0;
    for (int batch = 0; batch < num_batches; ++batch) //num_batches; ++batch)
    {
      /*
       * Possible implementation:
       * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
       * 2. compute each sub-batch of images' contribution to network
       * coefficient updates
       * 3. reduce the coefficient updates and broadcast to all nodes with
       * `MPI_Allreduce()'
       * 4. update local network coefficient at each node
       */


      nn_real adjusted_reg = reg/num_procs;
      int last_col = batch_start + get_batch_size(N, batch_size, batch);

      data_batch_size = last_col - batch_start;
      data_minibatch_size = data_batch_size / num_procs; 
      batch_start = last_col;

      // dnn.toGPU(nn); 

      // int  err = cudaMemcpy(dnn.d_gradW1, nn.W[0].memptr(), H0 * H1 * sizeof(nn_real), cudaMemcpyHostToDevice);
      // if (err != cudaSuccess)
      // {
      //   std::cerr << "Could not copy gradW1 to GPU" << std::endl;
      //   exit(1);
      // }


      // err = cudaMemcpy(dnn.d_gradW2, nn.W[1].memptr(), H1 * H2 * sizeof(nn_real), cudaMemcpyHostToDevice);
      // if (err != cudaSuccess)
      // {
      //   std::cerr << "Could not copy gradW2 to GPU" << std::endl;
      //   exit(1);
      // }



      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                          FEED FORWARD                            //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

      parallel_feedforward(dnn, dX_minibatch[batch], data_minibatch_size);


      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                         BACK PROPAGATE                           //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

    
      parallel_backprop(dnn,dX_minibatch[batch],dy_minibatch[batch],
      adjusted_reg, data_minibatch_size, data_batch_size);


      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    GRADIENT DESCENT STEP                         //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

      // gradients reduce gathering
      arma::Mat<nn_real> dW1(H1,H0);
      arma::Mat<nn_real> dW2(H2,H1);
      arma::Mat<nn_real> db1(H1,1);
      arma::Mat<nn_real> db2(H2,1);

      
      #define NEW_TRY 1
      #define NOT_NEW_TRY 0

    #if NOT_NEW_TRY
      dnn.fromGPU(h_gradW1,h_gradW2, h_gradb1,h_gradb2);
      arma::Mat<nn_real> mydW1 = arma::Mat<nn_real>(h_gradW1, H1, H0, true);
      arma::Mat<nn_real> mydW2 = arma::Mat<nn_real>(h_gradW2, H2, H1, true);
      arma::Mat<nn_real> mydb1 = arma::Mat<nn_real>(h_gradb1, H1, 1, true);
      arma::Mat<nn_real> mydb2 = arma::Mat<nn_real>(h_gradb2, H2, 1, true);

      if (num_procs>1){
       MPI_SAFE_CALL(MPI_Allreduce(mydW1.memptr(),dW1.memptr(),nn.H[1]*nn.H[0], MPI_FP, MPI_SUM, MPI_COMM_WORLD));
       MPI_SAFE_CALL(MPI_Allreduce(mydW2.memptr(),dW2.memptr(),H2*H1, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
       MPI_SAFE_CALL(MPI_Allreduce(mydb1.memptr(),db1.memptr(),H1*1, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
       MPI_SAFE_CALL(MPI_Allreduce(mydb2.memptr(),db2.memptr(),H2*1, MPI_FP, MPI_SUM, MPI_COMM_WORLD)); 
      nn.W[0] -= learning_rate * dW1;
      nn.W[1] -= learning_rate * dW2;
      nn.b[0] -= learning_rate * db1;
      nn.b[1] -= learning_rate * db2;
      }
      else{
        nn.W[0] -= learning_rate * mydW1;
        nn.W[1] -= learning_rate * mydW2;
        nn.b[0] -= learning_rate * mydb1;
        nn.b[1] -= learning_rate * mydb2;
      }
    #endif

    #if NEW_TRY

      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, dnn.d_gradW1,nn.H[1]*nn.H[0], MPI_FP, MPI_SUM, MPI_COMM_WORLD)); //  send rcv
      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, dnn.d_gradW2, H2*H1, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, dnn.d_gradb1,H1*1, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, dnn.d_gradb2,H2*1, MPI_FP, MPI_SUM, MPI_COMM_WORLD)); 

      myLRDifference(dnn.d_W1, dnn.d_gradW1 , nn.H[1], nn.H[0], learning_rate);
      myLRDifference(dnn.d_W2, dnn.d_gradW2 , nn.H[2], nn.H[1], learning_rate);
      myLRDifference(dnn.d_b1, dnn.d_gradb1 , nn.H[1], 1, learning_rate);
      myLRDifference(dnn.d_b2, dnn.d_gradb2 , H2, 1, learning_rate);

      // dnn.d_gradW1 = dnn.d_W1.copy();
      // dnn.d_gradW2 = dnn.d_W2.copy();
      // dnn.d_gradb1 = dnn.d_b1.copy();
      // dnn.d_gradb2 = dnn.d_b2.copy();


      // cudaMemcpy(nn.W[0].memptr(), dnn.d_W1, H1 * H0 * sizeof(nn_real), cudaMemcpyDeviceToHost); //  src, dst
      // cudaMemcpy(nn.W[1].memptr(), dnn.d_W2, H2 * H1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
      // cudaMemcpy(nn.b[0].memptr(), dnn.d_b1, H1 * 1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
      // cudaMemcpy(nn.b[1].memptr(), dnn.d_b2, H2 * 1 * sizeof(nn_real), cudaMemcpyDeviceToHost);

    #endif


      

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    POST-PROCESS OPTIONS                          //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      if (print_every <= 0)
      {
        print_flag = batch == 0;
      }
      else
      {
        print_flag = iter % print_every == 0;
      }

      /* Following debug routine assumes that you have already updated the arma
         matrices in the NeuralNetwork nn.  */
      // if (debug && rank == 0 && print_flag)
      // {
      //   // TODO
      //   // Copy data back to the CPU

      //   /* The following debug routine assumes that you have already updated the
      //    arma matrices in the NeuralNetwork nn.  */

      //   save_gpu_error(nn, iter, error_file);
      // }

      iter++;
    }
  }

  // MPI_Allreduce(&sendbuf,&recvbuf,count,datatype,op,comm);

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                  Update Neural Network on CPU                    //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                    Free memory allocations                       //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  // for( int batch = 0; batch < num_batches; ++batch){
  //   cudaFree(dX_minibatch[batch]);
  //   cudaFree(dy_minibatch[batch]);
  // }

  cudaMemcpy(nn.W[0].memptr(), dnn.d_W1, H1 * H0 * sizeof(nn_real), cudaMemcpyDeviceToHost); //  src, dst
  cudaMemcpy(nn.W[1].memptr(), dnn.d_W2, H2 * H1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[0].memptr(), dnn.d_b1, H1 * 1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[1].memptr(), dnn.d_b2, H2 * 1 * sizeof(nn_real), cudaMemcpyDeviceToHost);


  // free(h_gradW1);
  // free(h_gradW2);
  // free(h_gradb1);
  // free(h_gradb2);
  // free(h_Xbatch);
  // free(h_Ybatch);
  // free(h_yc);


}
