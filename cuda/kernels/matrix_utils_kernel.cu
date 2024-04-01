#include <cstddef>
#include <cuda_runtime.h>

namespace MatrixUtils {

	template <typename T>
	__global__ void bmm_kernel(const T *A, const T *B, T *C, int batch, int M, int K, int P) {
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

}

