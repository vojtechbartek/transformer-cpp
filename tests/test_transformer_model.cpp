#include "../transformer-cpp/transformer_model.hpp"
#include <iostream>
#include <vector>

int main() {
	// Read YAML configuration file
	YAML::Node config = YAML::LoadFile("configs/config.yaml");
	// Create a transformer model
	Transformer model(config);
	
	int batch_size = config["batch_size"].as<int>();
	int seq_len = config["seq_len"].as<int>();
	int embedding_dim = config["embedding_dim"].as<int>();
	int vocab_size = config["vocab_size"].as<int>();

	// Create input tensor of shape (batch_size, seq_len, embedding_dim)
	// fill with ones
	
	std::vector<std::vector<std::vector<float>>> input_tensor(batch_size, std::vector<std::vector<float>>(seq_len, std::vector<float>(embedding_dim, 1.0)));

	// Test forward pass
	auto output = model.forward(input_tensor);
	std::cout << "Output shape: " << output.size() << " x " << output[0].size() << " x " << output[0][0].size() << std::endl;
	// Print first 1 x seq_len x 3 elements
	for (int i = 0; i < output[0].size(); i++) {
		for (int ii = 0; ii < 3; ii++) {
			std::cout << output[0][i][ii] << " ";
		}
		std::cout << std::endl;
	}
	
	float sum = 0;
	bool test_passed = true;
	// Sum all rows and check, if they sum up to 1 (softmax)
	for (int b = 0; b < batch_size; b++) {
		for (int s = 0; s < seq_len; s++) {
			sum = 0;
			for (int v = 0; v < vocab_size; v++) {
				sum += output[b][s][v];
			}
			if (std::abs(sum - 1) > 1e-4) {
				std::cout << "[-] Forward sum test failed" << std::endl;
				std::cout << "Expected sum: 1" << std::endl;
				std::cout << "Got sum: " << sum << " at position " << b << " x " << s << std::endl;
				test_passed = false;
				break;
			}
		}
	}
	if (test_passed) {
		std::cout << "[+] Forward sum test passed" << std::endl;
	}
	
	
	if (output.size() != batch_size || output[0].size() != seq_len || output[0][0].size() != vocab_size) {
		std::cout << "[-] Forward shape test failed" << std::endl;
		std::cout << "Expected shape: " << batch_size << " x " << seq_len << " x " << vocab_size << std::endl;
		std::cout << "Got shape: " << output.size() << " x " << output[0].size() << " x " << output[0][0].size() << std::endl;
	}
	else {
		std::cout << "[+] Forward shape test passed" << std::endl;
	}

	// Test backward pass
	
	// Generate random grad_output
	std::vector<std::vector<std::vector<float>>> grad_output_tensor(batch_size, std::vector<std::vector<float>>(seq_len, std::vector<float>(vocab_size, 0.6)));

	auto grad_output = model.backward(grad_output_tensor, output);
	
	// Print first 1x seq_len x3 elements
	for (int i = 0; i < output[0].size(); i++) {
		for (int ii = 0; ii < 3; ii++) {
			std::cout << grad_output[0][i][ii] << " ";
		}
		std::cout << std::endl;
	}
	
	std::cout << "Grad output shape: " << grad_output.size() << " x " << grad_output[0].size() << " x " << grad_output[0][0].size() << std::endl;
	if (grad_output.size() != batch_size || grad_output[0].size() != seq_len || grad_output[0][0].size() != embedding_dim) {
		std::cout << "[-] Backward test failed" << std::endl;
		std::cout << "Expected shape: " << batch_size << " x " << seq_len << " x " << vocab_size << std::endl;
		std::cout << "Got shape: " << grad_output.size() << " x " << grad_output[0].size() << " x " << grad_output[0][0].size() << std::endl;
	}
	else {
		std::cout << "[+] Backward test passed" << std::endl;
	}


	return 0;
}
	
