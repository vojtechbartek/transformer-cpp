#include "../transformer-cpp/matrix_utils.hpp"
#include <iostream>
#include <vector>


int main() {

	std::cout << "Running tests for matrix_utils" << std::endl;

	std::vector<std::vector<int> > matrix1 = {
		{1, 2, 3},
		{4, -2, 6},
		{7, 0, 9},
		{1, 2, 1}
	};

	std::vector<std::vector<int> > matrix2 = {
		{1, 0, 2, 1},
		{1, 2, 3, 2},
		{0, 1, 3, -2}
	};

	std::vector<std::vector<int> > expected_result = {
		{3, 7, 17, -1},
		{2, 2, 20, -12},
		{7, 9, 41, -11},
		{3, 5, 11, 3}
	};
	// test matrix multiplication
	std::vector<std::vector<int>> result = MatrixUtils::matrixMultiplication(matrix1, matrix2);
	bool same = true;
	for (int i = 0; i < result.size(); i++) {
		for (int j = 0; j < result[i].size(); j++) {
			if (result[i][j] != expected_result[i][j]) {
				same = false;
				break;
			}
		}
	}
	if (same) {
		std::cout << "[+] Matrix multiplication test passed" << std::endl;
	} else {
		std::cout << "[-] Matrix multiplication test failed" << std::endl;
	}

	// test matrix transpose
	std::vector<std::vector<float>> matrix3 = {
		{1.2f, 2.0f},
		{4.1f, -2.2f},
		{7.9f, 0.0f}
	};

	std::vector<std::vector<float>> expected_transposed = {
		{1.2f, 4.1f, 7.9f},
		{2.0f, -2.2f, 0.0f}
	};

	std::vector<std::vector<float>> transposed = MatrixUtils::matrixTranspose(matrix3);
	if (transposed == expected_transposed) {
		std::cout << "[+] Matrix transpose test passed" << std::endl;
	} else {
		std::cout << "[-] Matrix transpose test failed" << std::endl;
	}	

	// test matrix softmax
	std::vector<std::vector<float>> matrix4 = {
		{1.2f, 2.0f},
		{4.1f, -2.2f},
		{7.9f, 0.0f}
	};
	
	std::vector<std::vector<float>> result_softmax = MatrixUtils::rowSoftmax(matrix4);
	bool softmaxed = true;
	for (int i = 0; i < result_softmax.size(); i++) {
		float sum = 0.0f;
		for (int j = 0; j < result_softmax[i].size(); j++) {
			// check if the result is between 0 and 1
			if (result_softmax[i][j] < 0.0f || result_softmax[i][j] > 1.0f) {
				softmaxed = false;
				std::cout << "[-] Matrix softmax test failed, value out of range [0,1] : " << result_softmax[i][j] << std::endl;
				break;
			}
			sum += result_softmax[i][j];
		}
		// check if the sum of the row is 1 with some tolerance
		if (std::abs(sum - 1.0f) > 0.0001f) {
			softmaxed = false;
			std::cout << "[-] Matrix softmax test failed, sum of row is not 1 : " << sum << std::endl;
			break;
		}
		
	}
	if (softmaxed) {
		std::cout << "[+] Matrix softmax test passed" << std::endl;
	} else {
		std::cout << "[-] Matrix softmax test failed" << std::endl;
	}

	// test matrix addition
	std::vector<std::vector<float>> matrix5 = {
		{1.0f, 2.1f, 3},
		{4, -2, 6},
		{7, 0, 9},
		{1, 2, 10000}
	};

	std::vector<std::vector<float>> matrix6 = {
		{1, 0.0f, 2},
		{1, 2.0f, 3},
		{0.1f, 1, 3.0f},
		{1, 2, 1}
	};

	std::vector<std::vector<float>> expected_result_addition = {
		{2.0f, 2.1f, 5},
		{5, 0, 9},
		{7.1f, 1, 12},
		{2, 4, 10001}
	};
	
	std::vector<std::vector<float>> result_addition = MatrixUtils::matrixAddition(matrix5, matrix6);
	bool same_addition = true;
	for (int i = 0; i < result_addition.size(); i++) {
		for (int j = 0; j < result_addition[i].size(); j++) {
			if (result_addition[i][j] != expected_result_addition[i][j]) {
				same_addition = false;
				break;
			}
		}
	}
	if (same_addition) {
		std::cout << "[+] Matrix addition test passed" << std::endl;
	} else {
		std::cout << "[-] Matrix addition test failed" << std::endl;
	}
	return 0;
}
