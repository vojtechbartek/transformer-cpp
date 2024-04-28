#include "../transformer-cpp/matrix_utils.hpp"
#include <iostream>
#include <vector>

float EPS = 1e-5;


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
		if (std::abs(sum - 1.0f) > EPS) {
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

	// test batch matrix mean
	// initialize tesnor of shape (2, 3, 2)
	std::vector<std::vector<std::vector<float>>> tensor = {
		{
			{1.0f, 2.1f},
			{4, -2},
			{7, 0}
		},
		{
			{1, 2},
			{4, -2},
			{3, 0}
		}
	};

	std::vector<std::vector<float>> expected_result_mean = {
		{1.0f, 2.05f},
		{4, -2},
		{5, 0}
	};

	std::vector<std::vector<float>> result_mean = MatrixUtils::batchMatrixMean(tensor);
	bool same_mean = true;
	for (int i = 0; i < result_mean.size(); i++) {
		for (int j = 0; j < result_mean[i].size(); j++) {
			if (result_mean[i][j] != expected_result_mean[i][j]) {
				same_mean = false;
				break;
			}
		}
	}
	if (same_mean) {
		std::cout << "[+] Batch matrix mean test passed" << std::endl;
	} else {
		std::cout << "[-] Batch matrix mean test failed" << std::endl;
	}

	// test batch vector mean
	// use the same tensor as above
	std::vector<float> expected_result_mean_vector = {10.0f / 3.0f, 0.05f / 3.0f};
	std::vector<float> result_mean_vector = MatrixUtils::batchVectorMean(tensor);
	bool same_mean_vector = true;
	for (int i = 0; i < result_mean_vector.size(); i++) {
		if (std::abs(result_mean_vector[i] - expected_result_mean_vector[i]) > EPS) {
			same_mean_vector = false;
			break;
		}
	}
	if (same_mean_vector) {
		std::cout << "[+] Batch vector mean test passed" << std::endl;
	} else {
		std::cout << "[-] Batch vector mean test failed" << std::endl;
	}

	std::vector<std::vector<float>> grad_output = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };
    std::vector<std::vector<float>> softmax_output = {
        {0.3543f, 0.6457f},
        {0.5498f, 0.4502f}
    };

    // Expected matrix
    std::vector<std::vector<float>> expected_softmax_der_output = {
        {-0.2288f, 0.2288f},
        {-0.2475f, 0.2475f}
    };
	std::vector<std::vector<float>> result_softmax_der = MatrixUtils::rowSoftmaxDerivative(grad_output, softmax_output);
	bool same_softmax_der = true;
	// Check dimensions
	if (result_softmax_der.size() != expected_softmax_der_output.size() || result_softmax_der[0].size() != expected_softmax_der_output[0].size()) {
		std::cout << "Softmax derivative test dimensions do not match" << std::endl;
		std::cout << "Expected : " << expected_softmax_der_output.size() << "x" << expected_softmax_der_output[0].size() << " got : " << result_softmax_der.size() << "x" << result_softmax_der[0].size() << std::endl;

		same_softmax_der = false;
	}
	for (int i = 0; i < result_softmax_der.size(); i++) {
		for (int j = 0; j < result_softmax_der[i].size(); j++) {
			if (std::abs(result_softmax_der[i][j] - expected_softmax_der_output[i][j]) > EPS) {
				same_softmax_der = false;
				std::cout << "at position (" << i << ", " << j << ") expected : " << expected_softmax_der_output[i][j] << " got : " << result_softmax_der[i][j] << std::endl;
				break;
			}
		}
	}
	if (same_softmax_der) {
		std::cout << "[+] Softmax derivative test passed" << std::endl;
	} else {
		std::cout << "[-] Softmax derivative test failed" << std::endl;
	}


	return 0;
}
