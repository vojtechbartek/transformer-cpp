#pragma once
#include <vector>
#include "feed_forward_layer.hpp"
#include "self_attention.hpp"

class TransformerLayer {
public:
	TransformerLayer(int input_dim, int embedding_dim, int head_size, int ff_hidden_dim, int output_dim);
	std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input);
	std::vector<std::vector<std::vector<float>>> backward(const std::vector<std::vector<std::vector<float>>>& grad_output);

private:
	SelfAttention self_attention;
	FeedForwardLayer feed_forward;
	std::vector<std::vector<float>> positional_embeddings;

	void apply_positional_encoding(std::vector<std::vector<std::vector<float>>> &input);

	int input_dim_, embedding_dim_, head_size_, ff_hidden_dim_, output_dim_;
	std::vector<std::vector<std::vector<float>>> input_, self_attention_input_, feed_forward_input_;
};
