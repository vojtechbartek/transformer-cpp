#pragma once

#include <vector>
#include <algorithm>

namespace ActivationFunctions {
	// ReLU activation function
	std::vector<std::vector<std::vector<float>>> ReLU(const std::vector<std::vector<std::vector<float>>>& input) {
	/*
	* ReLU activation function for a 3D tensor of shape (batch_size, seq_len, n)
	* input: 3D tensor of shape (batch_size, seq_len, n)
	*
	* output: 3D tensor of shape (batch_size, seq_len, n)
	*/
	
	int batch_size = input.size();
	int seq_len = input[0].size();
	int n = input[0][0].size();
	
	std::vector<std::vector<std::vector<float>>> output(batch_size, std::vector<std::vector<float>>(seq_len, std::vector<float>(n, 0.0)));

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < seq_len; j++) {
			for (int k = 0; k < n; k++) {
				output[i][j][k] = std::max(0.0f, input[i][j][k]);
			}
		}
	}

	return output;
	}

	std::vector<std::vector<std::vector<float>>> ReLU_derivative(const std::vector<std::vector<std::vector<float>>>& input) {
	/*
	* ReLU derivative activation function for a 3D tensor of shape (batch_size, seq_len, n)
	* input: 3D tensor of shape (batch_size, seq_len, n)
	*
	* output: 3D tensor of shape (batch_size, seq_len, n)
	*/

	int batch_size = input.size();
	int seq_len = input[0].size();
	int n = input[0][0].size();

	std::vector<std::vector<std::vector<float>>> output(batch_size, std::vector<std::vector<float>>(seq_len, std::vector<float>(n, 0.0)));

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < seq_len; j++) {
			for (int k = 0; k < n; k++) {
				output[i][j][k] = input[i][j][k] > 0 ? 1 : 0;
			}
		}
	}

	return output;
	}


};

