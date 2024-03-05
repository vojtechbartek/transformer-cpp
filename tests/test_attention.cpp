#include "../transformer-cpp/self_attention.hpp"
#include <iostream>

int main() {
	// test forward shape
	int batch_size = 2;
	int seq_len = 5;
	int embed_dim = 64;
	// create dummy input of shape (batch_size, seq_len, embed_dim)
	std::vector<std::vector<std::vector<float>>> input(batch_size, std::vector<std::vector<float>>(seq_len, std::vector<float>(embed_dim, 1.0f))); // Dummy input
	std::vector<std::vector<std::vector<float>>> mask(batch_size, std::vector<std::vector<float>>(seq_len, std::vector<float>(embed_dim, 1.0f))); // Dummy mask
	
	SelfAttention selfAttention(embed_dim, 1);
	std::vector<std::vector<std::vector<float>>> output = selfAttention.forward(input, mask);
	
	if (output.size() != batch_size || output[0].size() != embed_dim) {
		std::cerr << "[-] Self-attention Forward shape test failed" << std::endl;
		std::cerr << "Expected: (" << batch_size << ", " << seq_len << ", " << embed_dim << ")" << std::endl;
		std::cerr << "Got: (" << output.size() << ", " << output[0].size() << ", " << output[0][0].size() << ")" << std::endl;
	} else {
		std::cout << "[+] Self-attention Forward shape test passed" << std::endl;
	}

	// test masking
	mask[0][0][1] = 0.0f;
	output = selfAttention.forward(input, mask);
	std::vector<std::vector<std::vector<float>>> softmax_output = selfAttention.getSoftmaxOutput();
	float weight_at_masked_position = softmax_output[0][0][1];
	if (weight_at_masked_position >= 1e-6) {
		std::cerr << "[-] Self-attention Masking test failed" << std::endl;
	} else {
		std::cout << "[+] Self-attention Masking test passed" << std::endl;
	}

	// test backward
	// create dummy grad_output of shape (batch_size, seq_len, embed_dim)
	std::vector<std::vector<std::vector<float>>> grad_output(batch_size, std::vector<std::vector<float>>(seq_len, std::vector<float>(embed_dim, 1.0f))); // Dummy grad_output
	std::vector<std::vector<std::vector<float>>> grad_input = selfAttention.backward(input, grad_output);
	if (grad_input.size() != batch_size || grad_input[0].size() != seq_len || grad_input[0][0].size() != embed_dim) {
		std::cerr << "[-] Self-attention Backward shape test failed" << std::endl;
		std::cerr << "Expected: (" << batch_size << ", " << seq_len << ", " << embed_dim << ")" << std::endl;
		std::cerr << "Got: (" << grad_input.size() << ", " << grad_input[0].size() << ", " << grad_input[0][0].size() << ")" << std::endl;
	} else {
		std::cout << "[+] Self-attention Backward shape test passed" << std::endl;
	}
	

    return 0;
}

