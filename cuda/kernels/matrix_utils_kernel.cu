#include <cstddef>
#include <cuda_runtime.h>

namespace Kernel {

template <typename t>
__global__ void mm_kernel(const t *a, const t *b, t *c, int m, int k, int p) {
  /*
   * kernel for matrix multiplication,
   * each thread computes one cell of the output matrix c
   *
   * @param a: flat array of matrix of shape m x k
   * @param b: flat array of matrix of shape k x p
   * @param c: flat array of matrix of shape m x p, result of a x b
   */
  size_t x = blockidx.x * blockdim.x + threadidx.x;
  size_t y = blockidx.y * blockdim.y + threadidx.y;

  if ((x >= m) || (y >= p)) {
    return;
  }

  t sum = 0;

  for (size_t k = 0; k < k; k++) {
    sum += a[x * k + k] * b[k * p + y];
  }
  c[x * p + y] = sum;
}

template <typename T>
__global__ void mm_kernel(const T *A, const T *B, T *C, int batch, int M, int K,
                          int P) {
  /*
   * Kernel for batched matrix multiplication,
   * each thread computes one cell of the output matrix C
   *
   * @param A: flat array of matrix of shape batch x M x K
   * @param B: flat array of matrix of shape batch x K x P
   * @param C: flat array of matrix of shape batch x M x P, result of A x B
   */
  size_t b = blockIdx.z;
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= M) || (y >= P)) {
    return;
  }

  T sum = 0;

  for (size_t k = 0; k < K; k++) {
    sum += A[b * M * K + x * K + k] * B[b * K * P + k * P + y];
  }
  C[b * M * P + x * P + y] = sum;
}

template <typename T, int TILE_SIZE>
__global__ void mm_kernel_faster(const T *A, const T *B, T *C, int M, int K,
                                 int P) {
  /*
   * Kernel for matrix multiplication with shared memory, tile decomposition
   * and loop unrolling
   * 
   * @param A: flat array of matrix of shape M x K
   * @param B: flat array of matrix of shape K x P
   * @param C: flat array of matrix of shape M x P, result of A x B
   */

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  T value = 0;

  __shared__ T As[TILE_SIZE][TILE_SIZE];
  __shared__ T Bs[TILE_SIZE][TILE_SIZE];

  // Loop over the A and B tiles to compute the C element
  for (int t = 0; t < (K - 1) / TILE_SIZE + 1; t++) {
    // Load the tiles into shared memory
    if (row < M && t * TILE_SIZE + tx < K) {
      As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
    } else {
      As[ty][tx] = 0;
    }

    if (col < P && t * TILE_SIZE + ty < K) {
      Bs[ty][tx] = B[(t * TILE_SIZE + ty) * P + col];
    } else {
      Bs[ty][tx] = 0;
    }

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; i++) {
      value += As[ty][i] * Bs[i][tx];
    }

    __syncthreads();
  }
  
  // Write the final value to the output matrix
  if (row < M && col < P) {
    C[row * P + col] = value;
  }
}

template <typename T, int TILE_SIZE>
__global__ void mm_kernel_faster(const T *A, const T *B, T *C, int batch, int M, int K, int P) {
  /*
   * Optimized kernel for batched matrix multiplication using tiling and shared memory.
   * Each thread computes one cell of the output tensor C.
   *
   * @param A: flat array of matrix of shape batch x M x K
   * @param B: flat array of matrix of shape batch x K x P
   * @param C: flat array of matrix of shape batch x M x P, result of A x B
   */
  int b = blockIdx.z;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y; 
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  T value = 0;

  __shared__ T As[TILE_SIZE][TILE_SIZE];
  __shared__ T Bs[TILE_SIZE][TILE_SIZE];

  // loop over the tiles of A and B matrices to compute the C element
  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
    // load the tiles into shared memory
    if (row < M && t * TILE_SIZE + threadIdx.x < K) {
      As[threadIdx.y][threadIdx.x] = A[b * M * K + row * K + t * TILE_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0;
    }

    if (col < P && t * TILE_SIZE + threadIdx.y < K) {
      Bs[threadIdx.y][threadIdx.x] = B[b * K * P + (t * TILE_SIZE + threadIdx.y) * P + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; i++) {
      value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }

    __syncthreads();
  }

  // write the final value to the output matrix
  if (row < M && col < P) {
    C[b * M * P + row * P + col] = value;
  }
}



template <typename T>
__global__ void bmm_kernel(const T *A, const T *B, T *C, int batch, int M,
                           int K, int P) {
  /*
   * Kernel for matrix multiplication with broadcasted batch,
   * each thread computes one cell of the output matrix C
   *
   * @param A: flat array of matrix of shape batch x M x K
   * @param B: flat array of matrix of shape K x P
   * @param C: flat array of matrix of shape batch x M x P, result of
          applying A x B to each batch of A
   */
  size_t b = blockIdx.z;
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= M) || (y >= P)) {
    return;
  }

  T sum = 0;

  for (size_t k = 0; k < K; k++) {
    sum += A[b * M * K + x * K + k] * B[k * P + y];
  }
  C[b * M * P + x * P + y] = sum;
}

template <typename T, int TILE_SIZE>
__global__ void bmm_kernel_faster(const T *A, const T *B, T *C, int batch, int M,
                                  int K, int P) {
  /*
   * Optimized kernel for batched matrix multiplication with broadcasted batch using tiling and shared memory.
   * Each thread computes one cell of the output tensor C.
   *
   * @param A: flat array of matrix of shape batch x M x K
   * @param B: flat array of matrix of shape K x P
   * @param C: flat array of matrix of shape batch x M x P, result of applying A x B to each batch of A
   */

  int b = blockIdx.z;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  T value = 0;

  __shared__ T As[TILE_SIZE][TILE_SIZE];
  __shared__ T Bs[TILE_SIZE][TILE_SIZE];

  // loop over the tiles of A and B matrices to compute the C element
  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
    // load the tiles into shared memory
    if (row < M && t * TILE_SIZE + threadIdx.x < K) {
      As[threadIdx.y][threadIdx.x] = A[b * M * K + row * K + t * TILE_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0;
    }

    if (col < P && t * TILE_SIZE + threadIdx.y < K) {
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * P + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; i++) {
      value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }

    __syncthreads();
  }

  // write the final value to the output matrix
  if (row < M && col < P) {
    C[b * M * P + row * P + col] = value;
  }
}


template <typename T>
__global__ void matrix_add_kernel(const T *A, const T *B, T *C, int M, int N) {
  /*
   * Kernel for element-wise 2D matrix addition,
   * each thread computes one cell of the output matrix C
   * @param A: flat array of matrix of shape M x N
   * @param B: flat array of matrix of shape M x N
   * @param C: flat array of matrix of shape M x N, result of A + B
   * @param M: number of rows
   * @param N: number of columns
   */

  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= M) || (y >= N)) {
    return;
  }

  C[x * N + y] = A[x * N + y] + B[x * N + y];
}

template <typename T>
__global__ void matrix_add_kernel(const T *A, const T *B, T *C, int batch,
                                  int M, int N) {
  /*
   * Kernel for element-wise 3D matrix addition,
   * each thread computes one cell of the output matrix C
   * @param A: flat array of matrix of shape batch x M x N
   * @param B: flat array of matrix of shape batch x M x N
   * @param C: flat array of matrix of shape batch x M x N, result of A + B
   */
  size_t b = blockIdx.z;
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= M) || (y >= N)) {
    return;
  }

  C[b * M * N + x * N + y] =
      A[b * M * N + x * N + y] + B[b * M * N + x * N + y];
}

template <typename T>
__global__ void row_softmax_kernel(const T *A, T *B, int batch, int M, int N) {
  /*
   * Kernel for row-wise softmax operation,
   * each thread computes one row of the output matrix B
   * each row is computed by subtracting the maximum value of the row from each
   * element and then applying the softmax function which is defined as:
   * softmax(x) = exp(x) / sum(exp(x))
   *
   * @param A: flat array of matrix of shape batch x M x N
   * @param B: flat array of matrix of shape batch x M x N, result of softmax(A)
   */

  // TODO: very naive, maximum and sum are reduce operation and can be
  //	done in parallel by different tthreads

  size_t b = blockIdx.z;
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= M) {
    return;
  }

  T max_val = A[b * M * N +
                x * N]; // initialize max_val with the first element of the row
  // find the maximum value of the row
  for (size_t y = 1; y < N; y++) {
    if (A[b * M * N + x * N + y] > max_val) {
      max_val = A[b * M * N + x * N + y];
    }
  }

  T sum = 0;
  // compute the sum of the exponentials of the row elements
  for (size_t y = 0; y < N; y++) {
    B[b * M * N + x * N + y] = exp(A[b * M * N + x * N + y] - max_val);
    sum += B[b * M * N + x * N + y];
  }

  // divide each element by the sum to get the softmax
  for (size_t y = 0; y < N; y++) {
    B[b * M * N + x * N + y] /= sum;
  }
}

template <typename T>
__global__ void row_softmax_kernel(const T *A, T *B, int M, int N) {
  /*
   * Kernel for row-wise softmax operation,
   * each thread computes one row of the output matrix B
   * @param A: flat array of matrix of shape M x N
   * @param B: flat array of matrix of shape M x N, result of softmax(A)
   */

  size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= M) {
    return;
  }

  T max_val = A[x * N]; // initialize max_val with the first element of the row
  // find the maximum value of the row
  for (size_t y = 1; y < N; y++) {
    if (A[x * N + y] > max_val) {
      max_val = A[x * N + y];
    }
  }

  T sum = 0;

  // compute the sum of the exponentials of the row elements
  for (size_t y = 0; y < N; y++) {
    B[x * N + y] = exp(A[x * N + y] - max_val);
    sum += B[x * N + y];
  }

  // divide each element by the sum to get the softmax
  for (size_t y = 0; y < N; y++) {
    B[x * N + y] /= sum;
  }
}

template <typename T>
__global__ void row_softmax_derivative_kernel(const T *grad_output,
                                              const T *softmax_output,
                                              T *grad_softmax, int M, int N) {
  /**
   * Kernel for computing the derivative of the softmax function,
   * each thread computes one cell of the output matrix grad_softmax
   * @param grad_output: flat array of matrix of shape M x N, gradient of the
   * loss with respect to the softmax values
   * @param softmax_output: flat array of matrix of shape M x N, softmax output
   * @param grad_softmax: flat array of matrix of shape M x N, gradient of the
   * loss with respect to the input of the softmax function
   */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= M || j >= N) {
    return;
  }

  T diagonal_term = softmax_output[i * N + j] * (1 - softmax_output[i * N + j]);
  T off_diagonal_term = -softmax_output[i * N + j];
  T gradient = 0.0;

  for (int k = 0; k < N; k++) {
    gradient += grad_output[i * N + k] *
                (k == j ? diagonal_term
                        : off_diagonal_term * softmax_output[i * N + k]);
  }
  grad_softmax[i * N + j] = gradient;
}

template <typename T>
__global__ void row_softmax_derivative_kernel(T *grad_output, T *softmax_output,
                                              T *grad_softmax, int batch_size,
                                              int seq_len, int embed_dim) {
  /**
   * Kernel for computing the derivative of the softmax function,
   * each thread computes one cell of the output matrix grad_softmax
   * @param grad_output: flat array of matrix of shape batch_size x seq_len x
   * embed_dim, gradient of the loss with respect to the softmax values
   * @param softmax_output: flat array of matrix of shape batch_size x seq_len x
   * embed_dim, softmax output
   * @param grad_softmax: flat array of matrix of shape batch_size x seq_len x
   * embed_dim, gradient of the loss with respect to the input of the softmax
   * function
   * @param batch_size: number of sequences in the batch
   * @param seq_len: length of each sequence
   * @param embed_dim: dimension of the embedding
   */

  int b = blockIdx.z * blockDim.z + threadIdx.z;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (b < batch_size && i < seq_len && j < embed_dim) {
    int index_base = b * seq_len * embed_dim + i * embed_dim + j;
    T diagonal_term =
        softmax_output[index_base] * (1 - softmax_output[index_base]);
    T off_diagonal_term = -softmax_output[index_base];
    T gradient = 0.0;

    for (int k = 0; k < embed_dim; k++) {
      int k_index = b * seq_len * embed_dim + i * embed_dim + k;
      gradient += grad_output[k_index] *
                  (k == j ? diagonal_term
                          : off_diagonal_term * softmax_output[k_index]);
    }

    grad_softmax[index_base] = gradient;
  }
}

template <typename T>
__global__ void matrix_transpose_kernel(const T *A, T *B, int M, int N) {
  /*
   * Kernel for matrix transpose operation,
   * each thread computes one cell of the output matrix B
   * @param A: flat array of matrix of shape M x N
   * @param B: flat array of matrix of shape N x M, result of A^T
   */

  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= M) || (y >= N)) {
    return;
  }

  B[y * M + x] = A[x * N + y];
}

template <typename T>
__global__ void matrix_transpose_kernel(const T *A, T *B, int batch, int M,
                                        int N) {
  /*
   * Kernel for batched matrix transpose operation,
   * each thread computes one cell of the output matrix B
   * @param A: flat array of matrix of shape batch x M x N
   * @param B: flat array of matrix of shape batch x N x M, result of A^T
   */

  size_t b = blockIdx.z;
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= M) || (y >= N)) {
    return;
  }

  B[b * N * M + y * M + x] = A[b * M * N + x * N + y];
}

template <typename T>
__global__ void batch_vector_mean_kernel(const T *A, T *B, int batch, int M,
                                         int N) {
  /*
   * Kernel for computing the mean across the batch (first) and rows (second)
   * dimensions of a 3D tensor
   * @param A: flat array of matrix of shape batch x M x N
   * @param B: flat array of matrix of shape N, result of mean(A, axis=0,
   * axis=1)
   */

  size_t y = blockIdx.x * blockDim.x + threadIdx.x;

  if (y >= N) {
    return;
  }

  T sum = 0;
  // sum all the elements of the batch and rows
  // TODO: very naive, we can parallelize the sum operation and then divide by
  // the number of elements
  for (size_t b = 0; b < batch; b++) {
    for (size_t x = 0; x < M; x++) {
      sum += A[b * M * N + x * N + y];
    }
  }
  B[y] = sum / (batch * M);
}

template <typename T>
__global__ void batch_matrix_mean_kernel(const T *A, T *B, int batch, int M,
                                         int N) {
  /*
   * Kernel for computing the mean across the batch dimension of a 3D tensor
   * @param A: flat array of matrix of shape batch x M x N
   * @param B: flat array of matrix of shape M x N, result of mean(A, axis=0)
   */

  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= M) || (y >= N)) {
    return;
  }

  T sum = 0;
  // sum all the elements of the batch
  for (size_t b = 0; b < batch; b++) {
    sum += A[b * M * N + x * N + y];
  }
  B[x * N + y] = sum / batch;
}

} // namespace Kernel
