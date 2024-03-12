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
	if (output.size() != batch_size || output[0].size() != seq_len || output[0][0].size() != vocab_size) {
		std::cout << "[-] Forward test failed" << std::endl;
		std::cout << "Expected shape: " << batch_size << " x " << seq_len << " x " << vocab_size << std::endl;
		std::cout << "Got shape: " << output.size() << " x " << output[0].size() << " x " << output[0][0].size() << std::endl;
	}
	else {
		std::cout << "[+] Forward test passed" << std::endl;
	}
	
	// Test backward pass
	auto grad_output = model.backward(output);
	if (grad_output.size() != batch_size || grad_output[0].size() != seq_len || grad_output[0][0].size() != vocab_size) {
		std::cout << "[-] Backward test failed" << std::endl;
		std::cout << "Expected shape: " << batch_size << " x " << seq_len << " x " << vocab_size << std::endl;
		std::cout << "Got shape: " << grad_output.size() << " x " << grad_output[0].size() << " x " << grad_output[0][0].size() << std::endl;
	}
	else {
		std::cout << "[+] Backward test passed" << std::endl;
	}


	return 0;
}
	
