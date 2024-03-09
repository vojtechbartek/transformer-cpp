#include <iostream>
#include <vector>

#include "../transformer-cpp/feed_forward_layer.hpp"


int main() {
	int batch_size = 2;
	int seq_len = 3;
	int input_dim = 4;
	int hidden_dim = 10;
	int output_dim = 5;

	// test forward
	
	FeedForwardLayer forward_layer(input_dim, hidden_dim, output_dim);
	
	// initialize input
	std::vector<std::vector<std::vector<float>>> input = {
		{
			{1, 2.0, 1.1f, 0.5f},
			{0.5f, 1.0f, 0.5f, 0.1f},
			{0.1f, 0.2f, 0.1f, 0.05f}
		},
		{
			{0.1f, 0.2f, 0.1f, 0.05f},
			{0.5f, 1.0f, 0.5f, 0.1f},
			{1, 2.0, 1.1f, 0.5f}
		}
	};

	std::vector<std::vector<std::vector<float>>> output = forward_layer.forward(input);

	// check dimensions of the output
	if (output.size() != batch_size || output[0].size() != seq_len || output[0][0].size() != output_dim) {
		std::cerr << "Output dimensions are not correct" << std::endl;
		std::cerr << "Expected: " << batch_size << "x" << seq_len << "x" << output_dim << std::endl;
		std::cerr << "Got: " << output.size() << "x" << output[0].size() << "x" << output[0][0].size() << std::endl;
		return 1;
	}

	// print output
	/*
	for (int b = 0; b < batch_size; b++) {
		for (int i = 0; i < seq_len; i++) {
			for (int j = 0; j < output_dim; j++) {
				printf("%f ", output[b][i][j]);
			}
			printf("\n");
		}
	}
	*/

	std::cout << "[+] Forward test passed" << std::endl;

	// test backward
	
	// initialize grad_output
	std::vector<std::vector<std::vector<float>>> grad_output(batch_size, std::vector<std::vector<float>>(seq_len, std::vector<float>(output_dim, 0.5f)));
	
	std::vector<std::vector<std::vector<float>>> grad_input = forward_layer.backward(input, grad_output);

	// check dimensions of the grad_input
	if (grad_input.size() != batch_size || grad_input[0].size() != seq_len || grad_input[0][0].size() != input_dim) {
		std::cerr << "Grad input dimensions are not correct" << std::endl;
		std::cerr << "Expected: " << batch_size << "x" << seq_len << "x" << input_dim << std::endl;
		std::cerr << "Got: " << grad_input.size() << "x" << grad_input[0].size() << "x" << grad_input[0][0].size() << std::endl;
		return 1;
	}
	
	// check weight gradients dimensions
	std::vector<std::vector<std::vector<float>>> grad_weights = forward_layer.get_grad_weights();
	std::vector<std::vector<float>> grad_W1 = grad_weights[0];
	std::vector<std::vector<float>> grad_W2 = grad_weights[1];

	if (grad_W1.size() != input_dim || grad_W1[0].size() != hidden_dim) {
		std::cerr << "Grad W1 dimensions are not correct" << std::endl;
		std::cerr << "Expected: " << input_dim << "x" << hidden_dim << std::endl;
		std::cerr << "Got: " << grad_W1.size() << "x" << grad_W1[0].size() << std::endl;
		return 1;
	}

	if (grad_W2.size() != hidden_dim || grad_W2[0].size() != output_dim) {
		std::cerr << "Grad W2 dimensions are not correct" << std::endl;
		std::cerr << "Expected: " << hidden_dim << "x" << output_dim << std::endl;
		std::cerr << "Got: " << grad_W2.size() << "x" << grad_W2[0].size() << std::endl;
		return 1;
	}

	// check biases gradients dimensions
	std::vector<std::vector<float>> grad_biases = forward_layer.get_grad_biases();
	std::vector<float> grad_b1 = grad_biases[0];
	std::vector<float> grad_b2 = grad_biases[1];

	if (grad_b1.size() != hidden_dim) {
		std::cerr << "Grad b1 dimensions are not correct" << std::endl;
		std::cerr << "Expected: " << hidden_dim << std::endl;
		std::cerr << "Got: " << grad_b1.size() << std::endl;
		return 1;
	}
	
	if (grad_b2.size() != output_dim) {
		std::cerr << "Grad b2 dimensions are not correct" << std::endl;
		std::cerr << "Expected: " << output_dim << std::endl;
		std::cerr << "Got: " << grad_b2.size() << std::endl;
		return 1;
	}

	std::cout << "[+] Backward test passed" << std::endl;
	
	return 0;
}
	
	
