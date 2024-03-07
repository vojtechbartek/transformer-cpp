#pragma once

#include <vector>

class FeedForwardLayer {
public:
	FeedForwardLayer(int input_dim, int hidden_dim, int output_dim);
	std::vector<std::vector<std::vector<double>>> forward(std::vector<std::vector<std::vector<float>>>& input);

private:
	int input_dim, hidden_dim, output_dim;
	std::vector<std::vector<float>> W1, W2; // weights, not batched
	std::vector<float> b1, b2; // biases, not batched
    std::vector<std::vector<std::vector<float>>> add_bias(const std::vector<std::vector<std::vector<float>>>& input, const std::vector<float>& bias);
	void init_weights();
};
