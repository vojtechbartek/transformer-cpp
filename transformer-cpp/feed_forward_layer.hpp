#pragma once

#include <vector>

class FeedForwardLayer {
public:
  FeedForwardLayer(int input_dim, int hidden_dim, int output_dim);
  std::vector<std::vector<std::vector<float>>>
  forward(const std::vector<std::vector<std::vector<float>>> &input);
  std::vector<std::vector<std::vector<float>>>
  backward(const std::vector<std::vector<std::vector<float>>> &input,
           const std::vector<std::vector<std::vector<float>>> &grad_output);

  std::vector<std::vector<std::vector<float>>> get_grad_weights();
  std::vector<std::vector<float>> get_grad_biases();
  void update_weights(float learning_rate);

private:
  int input_dim, hidden_dim, output_dim;
  std::vector<std::vector<std::vector<float>>>
      hidden_relu; // output of ReLU, used in backward

  std::vector<std::vector<float>> W1, W2; // weights, not batched
  std::vector<float> b1, b2;              // biases, not batched

  std::vector<std::vector<float>> grad_W1,
      grad_W2;                         // gradients of weights, not batched
  std::vector<float> grad_b1, grad_b2; // gradients of biases, not batched

  std::vector<std::vector<std::vector<float>>>
  add_bias(const std::vector<std::vector<std::vector<float>>> &input,
           const std::vector<float> &bias);
  void init_weights();
};
