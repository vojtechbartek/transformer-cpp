#pragma once
#include "feed_forward_layer.hpp"
#include "self_attention.hpp"
#include <vector>

class TransformerLayer {
public:
  TransformerLayer(int input_dim, int embedding_dim, int head_size,
                   int ff_hidden_dim, int output_dim);
  std::vector<std::vector<std::vector<float>>>
  forward(const std::vector<std::vector<std::vector<float>>> &input);
  std::vector<std::vector<std::vector<float>>>
  backward(const std::vector<std::vector<std::vector<float>>> &grad_output);
  void update_weights(float learning_rate);

private:
  SelfAttention self_attention;
  FeedForwardLayer feed_forward;

  int input_dim_, embedding_dim_, head_size_, ff_hidden_dim_, output_dim_;
  std::vector<std::vector<std::vector<float>>> input_, self_attention_input_,
      feed_forward_input_;
};
