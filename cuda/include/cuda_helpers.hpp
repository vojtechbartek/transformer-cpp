#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>
#include "../kernels/matrix_utils_kernel.cu"
#include "cuda_config.hpp"



namespace CudaHelpers {
	template <typename T>
	void flatten3DMatrix(const std::vector<std::vector<std::vector<T>>>& matrix, T* flattened_matrix, int &size) {
		int batch_size = matrix.size();
		int seq_len = matrix[0].size();
		int embed_dim = matrix[0][0].size();	
		
		for (int b = 0; b < batch_size; b++) {
			for (int s = 0; s < seq_len; s++) {
				for (int e = 0; e < embed_dim; e++) {
					flattened_matrix[b * seq_len * embed_dim + s * embed_dim + e] = matrix[b][s][e];
				}
			}
		}
	}

	template <typename T>
	std::vector<std::vector<std::vector<T>>> unflatten3DMatrix(T* flattened_matrix, int batch_size, int seq_len, int embed_dim) {
		std::vector<std::vector<std::vector<T>>> matrix(batch_size, std::vector<std::vector<T>>(seq_len, std::vector<T>(embed_dim)));
		for (int b = 0; b < batch_size; b++) {
			for (int s = 0; s < seq_len; s++) {
				for (int e = 0; e < embed_dim; e++) {
					matrix[b][s][e] = flattened_matrix[b * seq_len * embed_dim + s * embed_dim + e];
				}
			}
		}
		return matrix;
	}

	template <typename T>
	std::vector<std::vector<std::vector<T>>> batchMatrixMultiplication(const std::vector<std::vector<std::vector<T>>>& matrix1, const std::vector<std::vector<std::vector<T>>>& matrix2) {
		/*
		* Helper function to perform batch matrix multiplication on the GPU
		* matrix1: 3D tensor of shape (batch_size, M, K) 
		* matrix2: 3D tensor of shape (batch_size, K, P)
		* Returns: 3D tensor of shape (batch_size, M, P)
		*/

		int batch_size = matrix1.size();
		int M = matrix1[0].size();
		int K = matrix1[0][0].size();
		int P = matrix2[0][0].size();
		assert(matrix2.size() == batch_size);
		assert(matrix2[0].size() == K);

		int size1 = batch_size * M * K;
		int size2 = batch_size * K * P;
		assert(size1 > 0);
		assert(size2 > 0);

		// Allocate gpu memory
		T *d_matrix1, *d_matrix2, *d_result;
		int result_size = batch_size * M * P;
		cudaMallocManaged(&d_matrix1, size1 * sizeof(T));
		cudaMallocManaged(&d_matrix2, size2 * sizeof(T));
		cudaMallocManaged(&d_result, result_size * sizeof(T));

		// Flatten the matrices
		flatten3DMatrix(matrix1, d_matrix1, size1);
		flatten3DMatrix(matrix2, d_matrix2, size2);
		
		// Call the kernel
		dim3 grid(1,1,1);
		dim3 threads(CudaConfig::BLOCK_SIZE, CudaConfig::BLOCK_SIZE);
		
		grid.z = batch_size;
		grid.x = std::ceil(static_cast<float>(M) / static_cast<float>(threads.x));
		grid.y = std::ceil(static_cast<float>(K) / static_cast<float>(threads.y));

		MatrixUtils::bmm_kernel<<<grid, threads>>>(d_matrix1, d_matrix2, d_result, batch_size, M, K, P);
		
		cudaError_t error = cudaPeekAtLastError();
		if (error != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
			std::exit(1);
		}

		cudaDeviceSynchronize();

		// Move the result from flat vector to 3D tensor
		std::vector<std::vector<std::vector<T>>> result = unflatten3DMatrix(d_result, batch_size, M, P);
		
		// Free the gpu memory
		cudaFree(d_matrix1);
		cudaFree(d_matrix2);
		cudaFree(d_result);
		
		return result;
		
	}
}
