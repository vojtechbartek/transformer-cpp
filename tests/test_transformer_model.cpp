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
	std::cout << "Output shape: " << output.size() << std::endl;
	return 0;
}
	
