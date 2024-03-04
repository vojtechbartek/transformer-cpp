#pragma once
#include <vector>

class SelfAttention {
public:
	SelfAttention(int embed_dim, int head_size);
	std::vector<std::vector<std::vector<float>>> forward(std::vector<std::vector<std::vector<float>>>& input,
						std::vector<std::vector<std::vector<float>>>& mask);	
	void updateParameters(float learning_rate);
	std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& grad_output); 
 
	std::vector<std::vector<float>> getWeights() const;
	std::vector<std::vector<float>> getSoftmaxOutput() const;

private:
	int embed_dim_;
	int head_size_;
	std::vector<std::vector<std::vector<float>>> softmax_output;
	std::vector<std::vector<std::vector<float>>> Q, K, V; // Query, key and value
	std::vector<std::vector<float>> Wq, Wk, Wv; // Weights for query, key and value
	std::vector<std::vector<float>> grad_Wq, grad_Wk, grad_Wv; // Gradients for query, key and value
};

