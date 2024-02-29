#pragma once
#include <vector>

class SelfAttention {
public:
	SelfAttention(int embed_dim, int num_heads);
	std::vector<std::vector<float>> forward(std::vector<std::vector<float>>& input,
						std::vector<std::vector<float>>& mask);	
    std::vector<std::vector<float>> backward(std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& grad_output);
	void updateParameters(float learning_rate);

	std::vector<std::vector<float>> getWeights() const;
	void setWeights(const std::vector<std::vector<float>>& weights);

private:
	int embed_dim_;
	int num_heads_;
	std::vector<std::vector<float>> Q, K, V; // Query, key and value
	std::vector<std::vector<float>> Wq, Wk, Wv; // Weights for query, key and value
	std::vector<std::vector<float>> grad_Wq, grad_Wk, grad_Wv; // Gradients for query, key and value
};

