#include <iostream>
#include <vector>
#include "../transformer-cpp/transformer_layer.hpp"

int main() {
	// Create a transformer layer
	int batch_size = 2;
	int seq_len = 3;
	int embed_dim = 4;
	int head_size = 2;
	int ff_hidden_dim = 8;
	int output_dim = 5;

	TransformerLayer transformer_layer(seq_len, embed_dim, head_size, ff_hidden_dim, output_dim);

	// Create input of shape (batch_size, seq_len, embed_dim)
	std::vector<std::vector<std::vector<float>>> input = {
	{
		{1.0f, 1.1f, 1.5f, 1.2f},
		{-2.0f, 2.1f, -2.5f, 2.2f},
		{3.0f, 3.1f, -3.5f, 3.2f}
	},
	{
		{4.0f, -4.1f, 4.5f, 4.2f},
		{5.0f, -5.1f, 5.5f, 5.2f},
		{-6.0f, 6.1f, 6.5f, -6.2f}
	}
	};
	
	// Forward pass
	std::vector<std::vector<std::vector<float>>> output = transformer_layer.forward(input);

	// Print output
	for (int i = 0; i < batch_size; i++) {
		std::cout << "Output for batch " << i << ":" << std::endl;
		for (int j = 0; j < seq_len; j++) {
			std::cout << "  ";
			for (int k = 0; k < output_dim; k++) {
				std::cout << output[i][j][k] << " ";
			}
			std::cout << std::endl;
		}
	}

	// check if output shape is as expected: batch_size x seq_len x output_dim
	if (output.size() == batch_size && output[0].size() == seq_len && output[0][0].size() == output_dim) {
		std::cout << "[+] Transformer layer output shape test passed" << std::endl; 
	} else {
		std::cout << "[-] Transformer layer output shape test failed" << std::endl;
		std::cout << "Expected: " << batch_size << " x " << seq_len << " x " << output_dim << std::endl;	
		std::cout << "Got: " << output.size() << " x " << output[0].size() << " x " << output[0][0].size() << std::endl;
	}

	// test backward pass
	std::vector<std::vector<std::vector<float>>> grad_output = {
	{
		{1.0f, 1.1f, 1.5f, 1.2f, 1.3f},
		{-2.0f, 2.1f, -2.5f, 2.2f, -2.3f},
		{2.0f, 1.1f, -1.5f, 2.2f, 3.3f}
	},
	{
		{1.0f, -2.1f, 1.5f, 1.2f, 2.3f},
		{3.0f, -1.1f, 1.5f, 2.2f, 2.3f},
		{-2.0f, 1.1f, 1.5f, -2.2f, -3.3f}
	}
	};

	std::vector<std::vector<std::vector<float>>> grad_input = transformer_layer.backward(grad_output);

	// Print grad_input
	for (int i = 0; i < batch_size; i++) {
		std::cout << "Grad input for batch " << i << ":" << std::endl;
		for (int j = 0; j < seq_len; j++) {
			std::cout << "  ";
			for (int k = 0; k < embed_dim; k++) {
				std::cout << grad_input[i][j][k] << " ";
			}
			std::cout << std::endl;
		}
	}

	// check if grad_input shape is as expected: batch_size x seq_len x embed_dim
	if (grad_input.size() == batch_size && grad_input[0].size() == seq_len && grad_input[0][0].size() == embed_dim) {
		std::cout << "[+] Transformer layer grad_input shape test passed" << std::endl; 
	} else {
		std::cout << "[-] Transformer layer grad_input shape test failed" << std::endl;
		std::cout << "Expected: " << batch_size << " x " << seq_len << " x " << embed_dim << std::endl;	
		std::cout << "Got: " << grad_input.size() << " x " << grad_input[0].size() << " x " << grad_input[0][0].size() << std::endl;
	}

	return 0;
}


		
