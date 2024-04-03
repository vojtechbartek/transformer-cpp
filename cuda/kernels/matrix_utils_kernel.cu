#include <cstddef>
#include <cuda_runtime.h>

namespace MatrixUtils {

	template <typename T>
	__global__ void mm_kernel(const T *A, const T *B, T *C, int M, int K, int P) {
		/*
		 * Kernel for matrix multiplication,
		 * each thread computes one cell of the output matrix C
		 *
		 * @param A: flat array of matrix of shape M x K
		 * @param B: flat array of matrix of shape K x P
		 * @param C: flat array of matrix of shape M x P, result of A x B
		 */
		size_t x = blockIdx.x * blockDim.x + threadIdx.x;
		size_t y = blockIdx.y * blockDim.y + threadIdx.y;
		
		if ((x >= M) || (y >= P)) {
			return;
		}

		T sum = 0;

		for (size_t k = 0; k < K; k++) {
			sum += A[x * K + k] * B[k * P + y];
		}
		C[x * P + y] = sum;
	}

	template <typename T>
	__global__ void mm_kernel(const T *A, const T *B, T *C, int batch, int M, int K, int P) {
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

	template <typename T>
	__global__ void bmm_kernel(const T *A, const T *B, T *C, int batch, int M, int K, int P) {
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
	__global__ void matrix_add_kernel(const T *A, const T *B, T *C, int batch, int M, int N) {
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
		
		C[b * M * N + x * N + y] = A[b * M * N + x * N + y] + B[b * M * N + x * N + y];
	}

	template <typename T>
	__global__ void row_softmax_kernel(const T *A, T *B, int batch, int M, int N) {
	/*
	* Kernel for row-wise softmax operation,
	* each thread computes one row of the output matrix B
	* each row is computed by subtracting the maximum value of the row from each element
	* and then applying the softmax function which is defined as:
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

	T max_val = A[b * M * N + x * N]; // initialize max_val with the first element of the row
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
	__global__ void row_softmax_derivative_kernel(const T *A, const T *B, T *C, int M, int N) {
	/*
	* Kernel for computing the derivative of the softmax function,
	* each thread computes one row of the output matrix C
	* @param A: flat array of matrix of shape M x N, softmax output
	* @param B: flat array of matrix of shape M x N, gradient of the loss with respect to the softmax values
	* @param C: flat array of matrix of shape M x N, gradient of the loss with respect to the input of the softmax function
	*/

	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x >= M) {
		return;
	}

	for (size_t y = 0; y < N; y++) {
		T index = x * N + y;
		T s_i = A[index];
		T grad = 0;
		for (size_t k = 0; k < N; k++) {
			T j_index = x * N + k;		
			if (k == y) {
				// diagonal part: softmax(x_i) * (1 - softmax(x_i))
				grad += s_i * (1 - s_i) * B[j_index];
			} else {
				// off-diagonal part: -softmax(x_i) * softmax(x_j)
				grad -= s_i * A[j_index] * B[j_index];
			}
		}
		C[index] = grad;
	}
	}

	template <typename T>
	__global__ void row_softmax_derivative_kernel(const T *A, const T *B, T *C, int batch, int M, int N) {
	/*
	* Kernel for computing the derivative of the softmax function,
	* each thread computes one row of the output matrix C
	* @param A: flat array of matrix of shape batch x M x N, softmax output
	* @param B: flat array of matrix of shape batch x M x N, gradient of the loss with respect to the softmax values
	* @param C: flat array of matrix of shape batch x M x N, gradient of the loss with respect to the input of the softmax function
	*/

	size_t b = blockIdx.z;
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= M) {
		return;
	}

	for (size_t y = 0; y < N; y++) {
		T index = b * M * N + x * N + y;
		T s_i = A[index];
		T grad = 0;
		for (size_t k = 0; k < N; k++) {
			T j_index = b * M * N + x * N + k;		
			if (k == y) {
				// diagonal part: softmax(x_i) * (1 - softmax(x_i))
				grad += s_i * (1 - s_i) * B[j_index];
			} else {
				// off-diagonal part: -softmax(x_i) * softmax(x_j)
				grad -= s_i * A[j_index] * B[j_index];
			}
		}
		C[index] = grad;
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
	__global__ void matrix_transpose_kernel(const T *A, T *B, int batch, int M, int N) {
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
	__global__ void batch_vector_mean(const T *A, T *B, int batch, int M, int N) {
	/*
	* Kernel for computing the mean across the batch (first) and rows (second) 
	* dimensions of a 3D tensor
	* @param A: flat array of matrix of shape batch x M x N
	* @param B: flat array of matrix of shape N, result of mean(A, axis=0, axis=1)
	*/

	size_t y = blockIdx.x * blockDim.x + threadIdx.x;

	if (y >= N) {
		return;
	}

	T sum = 0;
	// sum all the elements of the batch and rows
	// TODO: very naive, we can parallelize the sum operation and then divide by the number of elements
	for (size_t b = 0; b < batch; b++) {
		for (size_t x = 0; x < M; x++) {
			sum += A[b * M * N + x * N + y];
		}
	}
	B[y] = sum / (batch * M);
	}

	template <typename T>
	__global__ void batch_matrix_mean(const T *A, T *B, int batch, int M, int N) {
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
	

}

