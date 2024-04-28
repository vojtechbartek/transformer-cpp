#pragma once
#include "feed_forward_layer.hpp"
#include "positional_encoding.hpp"
#include "self_attention.hpp"
#include "transformer_layer.hpp"
#include <yaml-cpp/yaml.h>

class Transformer {
public:
  Transformer(const YAML::Node &config);
  std::vector<std::vector<std::vector<float>>>
  forward(const std::vector<std::vector<std::vector<float>>> &input);
  std::vector<std::vector<std::vector<float>>>
  backward(const std::vector<std::vector<std::vector<float>>> &grad_output,
           const std::vector<std::vector<std::vector<float>>> &output_logits);
  void update_weights(float learning_rate);

private:
  std::vector<TransformerLayer> layers;
  void apply_positional_encoding(
      std::vector<std::vector<std::vector<float>>> &input);
  std::vector<std::vector<float>> create_mask(int seq_length);

  std::vector<std::vector<float>> positional_embeddings;
  int num_layers;
  int embedding_dim;
  int batch_size;
  int seq_len;
  int vocab_size;
};
