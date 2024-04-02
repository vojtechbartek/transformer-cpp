#include "../cuda/include/cuda_helpers.hpp"
#include "../transformer-cpp/matrix_utils.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>


bool all_close(const std::vector<std::vector<std::vector<float>>> &a,
			   const std::vector<std::vector<std::vector<float>>> &b) {
	if (a.size() != b.size()) {
		return false;
	}

  for (size_t i = 0; i < a.size(); i++) {
	for (size_t j = 0; j < a[i].size(); j++) {
	  for (size_t k = 0; k < a[i][j].size(); k++) {
		if (std::abs(a[i][j][k] - b[i][j][k]) > 1e-6) {
		  return false;
		}
	  }
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

bool random_batch_matrix_multiplication_test(int b, int m, int k, int p) {
	std::vector<std::vector<std::vector<float>>> A = create_random_matrix(b, m, k);
	std::vector<std::vector<std::vector<float>>> B = create_random_matrix(b, k, p);

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
	

int main() {
	int b, m, k, p;	
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(1, 5);
	
	// Run 10 random tests for matrix multiplication
	for (int i = 0; i < 10; i++) {
		b = distribution(generator);
		m = distribution(generator);
		k = distribution(generator);
		p = distribution(generator);

		if (!random_batch_matrix_multiplication_test(b, m, k, p)) {
			std::cout << "[-] Matrix Multiplication Test failed" << std::endl;
			return 1;
		}
	}	
	std::cout << "[+] Matrix Multiplication Test passed" << std::endl;
	
	// Run 10 random tests for matrix addition 2D
	for (int i = 0; i < 10; i++) {
		m = distribution(generator);
		k = distribution(generator);
		
		if (!random_matrix_addition2D_test(m, k)) {
			std::cout << "[-] Matrix Addition Test failed" << std::endl;
			return 1;
		}
	}

	return 0;

}
