#pragma once
#include <yaml-cpp/yaml.h>
#include "transformer_layer.hpp"
#include "positional_encoding.hpp"
#include "self_attention.hpp"
#include "feed_forward_layer.hpp"

class Transformer {
public:
	Transformer(const YAML::Node& config);
	std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input);
	void backward(const std::vector<std::vector<std::vector<float>>>& grad_output);
	void update_weights(float learning_rate);

private:
	std::vector<TransformerLayer> layers;
	void apply_positional_encoding(std::vector<std::vector<std::vector<float>>>& input);
	std::vector<std::vector<float>> create_mask(int seq_length);
	
	std::vector<std::vector<float>> positional_embeddings;
	int num_layers;
	int embedding_dim;
	int batch_size;
	int seq_len;
	int vocab_size;

};	
	
