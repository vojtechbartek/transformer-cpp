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

}

