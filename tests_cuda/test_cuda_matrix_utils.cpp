#include "../cuda/include/cuda_helpers.hpp"
#include "../transformer-cpp/matrix_utils.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

double EPSILON = 1e-6;

bool all_close(const std::vector<std::vector<std::vector<float>>> &a,
			   const std::vector<std::vector<std::vector<float>>> &b) {
	if ((a.size() != b.size()) || (a[0].size() != b[0].size()) || (a[0][0].size() != b[0][0].size()){
		return false;
	}

  for (size_t i = 0; i < a.size(); i++) {
	for (size_t j = 0; j < a[i].size(); j++) {
	  for (size_t k = 0; k < a[i][j].size(); k++) {
		if (std::abs(a[i][j][k] - b[i][j][k]) > EPSILON) {
		  return false;
		}
	  }
	}
  }
  return true;
}

bool all_close(const std::vector<std::vector<float>> &a,
			   const std::vector<std::vector<float>> &b) {
	if ((a.size() != b.size()) || (a[0].size() != b[0].size())) {
		return false;
	}

	for (size_t i = 0; i < a.size(); i++) {
		for (size_t j = 0; j < a[i].size(); j++) {
			if (std::abs(a[i][j] - b[i][j]) > EPSILON) {
				return false;
			}
		}
	}
  return true;
}

bool all_close(const std::vector<float> &a,
			   const std::vector<float> &b) {
	if (a.size() != b.size()) {
		return false;
	}

	for (size_t i = 0; i < a.size(); i++) {
		if (std::abs(a[i] - b[i]) > EPSILON) {
			return false;
		}
	}
  return true;
}

std::vector<std::vector<float>> create_random_matrix(int m, int k) {
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_real_distribution<float> distribution(-1.0, 1.0);
	
	std::vector<std::vector<float>> matrix(m, std::vector<float>(k));
	for (int j = 0; j < m; j++) {
		for (int l = 0; l < k; l++) {
			matrix[j][l] = distribution(generator);
		}
	}
	return matrix;
}

std::vector<std::vector<std::vector<float>>> create_random_matrix(int b, int m, int k) {
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_real_distribution<float> distribution(-1.0, 1.0);
	
	std::vector<std::vector<std::vector<float>>> matrix(b, std::vector<std::vector<float>>(m, std::vector<float>(k)));
	for (int i = 0; i < b; i++) {
		for (int j = 0; j < m; j++) {
			for (int l = 0; l < k; l++) {
				matrix[i][j][l] = distribution(generator);
			}
		}
	}
	return matrix;
}


void print_matrix(const std::vector<std::vector<std::vector<float>>> &matrix) {
	for (size_t i = 0; i < matrix.size(); i++) {
		for (size_t j = 0; j < matrix[i].size(); j++) {
			for (size_t k = 0; k < matrix[i][j].size(); k++) {
				std::cout << matrix[i][j][k] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

bool random_matrix_multiplication3D_test(int b, int m, int k, int p) {
	std::vector<std::vector<std::vector<float>>> A = create_random_matrix(b, m, k);
	std::vector<std::vector<std::vector<float>>> B = create_random_matrix(b, k, p);

	std::vector<std::vector<std::vector<float>>> C_cuda = CudaHelpers::batchMatrixMultiplication(A, B);
	std::vector<std::vector<std::vector<float>>> C_cpu = MatrixUtils::batchMatrixMultiplication(A, B);
		
	return all_close(C_cuda, C_cpu);
}

bool random_matrix_multiplication2D_test(int m, int k, int p) {
	std::vector<std::vector<float>> A = create_random_matrix(m, k);
	std::vector<std::vector<float>> B = create_random_matrix(k, p);

	std::vector<std::vector<float>> C_cuda = CudaHelpers::matrixMultiplication(A, B);
	std::vector<std::vector<float>> C_cpu = MatrixUtils::matrixMultiplication(A, B);

	return all_close(C_cuda, C_cpu);
}

bool random_batch_matrix_multiplication_test(int b, int m, int k, int p) {
	std::vector<std::vector<std::vector<float>>> A = create_random_matrix(b, m, k);
	std::vector<std::vector<float>> B = create_random_matrix(k, p);

	std::vector<std::vector<std::vector<float>>> C_cuda = CudaHelpers::batchMatrixMultiplication(A, B);
	std::vector<std::vector<std::vector<float>>> C_cpu = MatrixUtils::batchMatrixMultiplication(A, B);

	return all_close(C_cuda, C_cpu);
}

bool random_matrix_addition2D_test(int m, int n) {
	std::vector<std::vector<float>> A = create_random_matrix(m, n);
	std::vector<std::vector<float>> B = create_random_matrix(m, n);
	
	std::vector<std::vector<float>> C_cuda = CudaHelpers::matrixAddition(A, B);
	std::vector<std::vector<float>> C_cpu = MatrixUtils::matrixAddition(A, B);

	return all_close(C_cuda, C_cpu);
}

bool random_matrix_addition3D_test(int b, int m, int n) {
	std::vector<std::vector<std::vector<float>>> A = create_random_matrix(b, m, n);
	std::vector<std::vector<std::vector<float>>> B = create_random_matrix(b, m, n);
	
	std::vector<std::vector<std::vector<float>>> C_cuda = CudaHelpers::matrixAddition(A, B);
	std::vector<std::vector<std::vector<float>>> C_cpu = MatrixUtils::matrixAddition(A, B);

	return all_close(C_cuda, C_cpu);
}
	
bool random_row_softmax_test(int m, int n) {
	std::vector<std::vector<float>> A = create_random_matrix(m, n);
	std::vector<std::vector<float>> C_cuda = CudaHelpers::rowSoftmax(A);
	std::vector<std::vector<float>> C_cpu = MatrixUtils::rowSoftmax(A);

	return all_close(C_cuda, C_cpu);
}

bool random_batch_row_softmax_test(int b, int m, int n) {
	std::vector<std::vector<std::vector<float>>> A = create_random_matrix(b, m, n);
	std::vector<std::vector<std::vector<float>>> C_cuda = CudaHelpers::rowSoftmax(A);
	std::vector<std::vector<std::vector<float>>> C_cpu = MatrixUtils::rowSoftmax(A);

	return all_close(C_cuda, C_cpu);
}

bool random_row_softmax_derivative_test(int m, int n) {
	std::vector<std::vector<float>> softmax_input = create_random_matrix(m, n);
	std::vector<std::vector<float>> softmax_output = MatrixUtils::rowSoftmax(softmax_input);
	std::vector<std::vector<float>> grad_output = create_random_matrix(m, n);

	std::vector<std::vector<float>> C_cuda = CudaHelpers::rowSoftmaxDerivative(softmax_output, grad_output);
	std::vector<std::vector<float>> C_cpu = MatrixUtils::rowSoftmaxDerivative(softmax_output, grad_output);

	return all_close(C_cuda, C_cpu);
}

bool random_batch_row_softmax_derivative_test(int b, int m, int n) {
	std::vector<std::vector<std::vector<float>>> softmax_input = create_random_matrix(b, m, n);
	std::vector<std::vector<std::vector<float>>> softmax_output = MatrixUtils::rowSoftmax(softmax_input);
	std::vector<std::vector<std::vector<float>>> grad_output = create_random_matrix(b, m, n);

	std::vector<std::vector<std::vector<float>>> C_cuda = CudaHelpers::rowSoftmaxDerivative(softmax_output, grad_output);
	std::vector<std::vector<std::vector<float>>> C_cpu = MatrixUtils::rowSoftmaxDerivative(softmax_output, grad_output);

	return all_close(C_cuda, C_cpu);
}

bool random_matrix_transpose_test(int m, int n) {
	std::vector<std::vector<float>> A = create_random_matrix(m, n);
	std::vector<std::vector<float>> C_cuda = CudaHelpers::matrixTranspose(A);
	std::vector<std::vector<float>> C_cpu = MatrixUtils::matrixTranspose(A);

	return all_close(C_cuda, C_cpu);
}

bool random_batch_matrix_transpose_test(int b, int m, int n) {
	std::vector<std::vector<std::vector<float>>> A = create_random_matrix(b, m, n);
	std::vector<std::vector<std::vector<float>>> C_cuda = CudaHelpers::batchMatrixTranspose(A);
	std::vector<std::vector<std::vector<float>>> C_cpu = MatrixUtils::batchMatrixTranspose(A);

	return all_close(C_cuda, C_cpu);
}

bool random_batch_matrix_mean_test(int b, int m, int n) {
	std::vector<std::vector<std::vector<float>>> A = create_random_matrix(b, m, n);
	std::vector<std::vector<float>> C_cuda = CudaHelpers::batchMatrixMean(A);
	std::vector<std::vector<float>> C_cpu = MatrixUtils::batchMatrixMean(A);

	return all_close(C_cuda, C_cpu);
}

bool random_vector_mean_test(int b, int m, int n) {
	std::vector<std::vector<std::vector<float>>> A = create_random_matrix(b, m, n);
	std::vector<float> C_cuda = CudaHelpers::batchVectorMean(A);
	std::vector<float> C_cpu = MatrixUtils::batchVectorMean(A);

	return all_close(C_cuda, C_cpu);
}


int main() {
	int b, m, k, p;	
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(1, 30);
	
	int num_tests = 10;
		
	// Run num_tests random tests for batch matrix multiplication
	// 3D * 2D
	for (int i = 0; i < num_tests; i++) {
		b = distribution(generator);
		m = distribution(generator);
		k = distribution(generator);
		p = distribution(generator);

		if (!random_batch_matrix_multiplication_test(b, m, k, p)) {
			std::cout << "[-] Batch Matrix Multiplication Test failed" << std::endl;
			return 1;
		}
	}	
	std::cout << "[+] Batch Matrix Multiplication Test passed" << std::endl;
	
	// Run num_tests random tests for matrix multiplication 2D
	for (int i = 0; i < num_tests; i++) {
		m = distribution(generator);
		k = distribution(generator);
		p = distribution(generator);

		if (!random_matrix_multiplication2D_test(m, k, p)) {
			std::cout << "[-] Matrix Multiplication 2D Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Matrix Multiplication 2D Test passed" << std::endl;

	// Run num_tests random tests for matrix multiplication 3D
	for (int i = 0; i < num_tests; i++) {
		b = distribution(generator);
		m = distribution(generator);
		k = distribution(generator);
		p = distribution(generator);

		if (!random_matrix_multiplication3D_test(b, m, k, p)) {
			std::cout << "[-] Matrix Multiplication 3D Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Matrix Multiplication 3D Test passed" << std::endl;

	// Run num_tests random tests for matrix addition 2D
	for (int i = 0; i < num_tests; i++) {
		m = distribution(generator);
		k = distribution(generator);
		
		if (!random_matrix_addition2D_test(m, k)) {
			std::cout << "[-] Matrix Addition 2D Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Matrix Addition Test passed" << std::endl;

	// Run 10 random tests for matrix addition 3D
	for (int i = 0; i < 10; i++) {
		b = distribution(generator);
		m = distribution(generator);
		k = distribution(generator);
		
		if (!random_matrix_addition3D_test(b, m, k)) {
			std::cout << "[-] Matrix Addition 3D Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Matrix Addition 3D Test passed" << std::endl;

	// Run num_tests random tests for row-wise softmax 2D
	for (int i = 0; i < num_tests; i++) {
		m = distribution(generator);
		k = distribution(generator);
		
		if (!random_row_softmax_test(m, k)) {
			std::cout << "[-] Row-wise Softmax 2D Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Row-wise Softmax 2D Test passed" << std::endl;

	// Run num_tests random tests for row-wise softmax 3D
	for (int i = 0; i < num_tests; i++) {
		b = distribution(generator);
		m = distribution(generator);
		k = distribution(generator);
		
		if (!random_batch_row_softmax_test(b, m, k)) {
			std::cout << "[-] Row-wise Softmax 3D Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Row-wise Softmax 3D Test passed" << std::endl;

	// Run num_tests random tests for row-wise softmax derivative 2D
	for (int i = 0; i < num_tests; i++) {
		m = distribution(generator);
		k = distribution(generator);
		
		if (!random_row_softmax_derivative_test(m, k)) {
			std::cout << "[-] Row-wise Softmax Derivative 2D Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Row-wise Softmax Derivative 2D Test passed" << std::endl;

	// Run num_tests random tests for row-wise softmax derivative 3D
	for (int i = 0; i < num_tests; i++) {
		b = distribution(generator);
		m = distribution(generator);
		k = distribution(generator);
		
		if (!random_batch_row_softmax_derivative_test(b, m, k)) {
			std::cout << "[-] Row-wise Softmax Derivative 3D Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Row-wise Softmax Derivative 3D Test passed" << std::endl;

	// Run num_tests random tests for matrix transpose 2D
	for (int i = 0; i < num_tests; i++) {
		m = distribution(generator);
		k = distribution(generator);
		
		if (!random_matrix_transpose_test(m, k)) {
			std::cout << "[-] Matrix Transpose 2D Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Matrix Transpose 2D Test passed" << std::endl;

	// Run num_tests random tests for matrix transpose 3D
	for (int i = 0; i < num_tests; i++) {
		b = distribution(generator);
		m = distribution(generator);
		k = distribution(generator);
		
		if (!random_batch_matrix_transpose_test(b, m, k)) {
			std::cout << "[-] Matrix Transpose 3D Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Matrix Transpose 3D Test passed" << std::endl;

	// Run num_tests random tests for batch matrix mean
	for (int i = 0; i < num_tests; i++) {
		b = distribution(generator);
		m = distribution(generator);
		k = distribution(generator);
		
		if (!random_batch_matrix_mean_test(b, m, k)) {
			std::cout << "[-] Batch Matrix Mean Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Batch Matrix Mean Test passed" << std::endl;

	// Run num_tests random tests for vector mean
	for (int i = 0; i < num_tests; i++) {
		b = distribution(generator);
		m = distribution(generator);
		k = distribution(generator);
		
		if (!random_vector_mean_test(b, m, k)) {
			std::cout << "[-] Vector Mean Test failed" << std::endl;
			return 1;
		}
	}
	std::cout << "[+] Vector Mean Test passed" << std::endl;

	return 0;
}

