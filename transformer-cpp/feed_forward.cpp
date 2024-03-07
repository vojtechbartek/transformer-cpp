#include "feed_forward.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <vector>
#include <random>
#include <iostream>

FeedForwardLayer::FeedForwardLayer(int input_dim, int hidden_dim, int output_dim) : input_dim(input_dim), hidden_dim(hidden_dim), output_dim(output_dim) {
    init_weights();
}

void FeedForwardLayer::init_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 0.1); // mean = 0, std = 0.1
    
    W1 = std::vector<std::vector<float>>(input_dim, std::vector<float>(hidden_dim));
    W2 = std::vector<std::vector<float>>(hidden_dim, std::vector<float>(output_dim));
    b1 = std::vector<float>(hidden_dim, 0);
    b2 = std::vector<float>(output_dim, 0);

    for (int i = 0; i < input_dim; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            W1[i][j] = dist(gen);
        }
    }

    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < output_dim; j++) {
            W2[i][j] = dist(gen);
        }
    }
}

std::vector<std::vector<std::vector<float>>> FeedForwardLayer::forward(const std::vector<std::vector<std::vector<float>>>& input) {
  /*
    * input: batch_size x seq_len x head_size
    * 
    * output: batch_size x seq_len x output_dim
    */

  std::vector<std::vector<std::vector<float>>> hidden = add_bias(MatrixUtils::batchMatrixMultiplication(input, W1), b1); // batch_size x seq_len x hidden_dim
  std::vector<std::vector<std::vector<float>>> hidden_relu = ActivationFunctions::relu(hidden); // batch_size x seq_len x hidden_dim
  std::vector<std::vector<std::vector<float>>> output = add_bias(MatrixUtils::batchMatrixMultiplication(hidden_relu, W2), b2); // batch_size x seq_len x output_dim

  return output;
}

std::vector<std::vector<std::vector<float>>> FeedForwardLayer::add_bias(const std::vector<std::vector<std::vector<float>>>& input, const std::vector<float>& bias) {
  std::vector<std::vector<std::vector<float>>> output = input;
  for (int b = 0; b < input.size(); ++b) {
      for (int i = 0; i < input[b].size(); ++i) {
          for (int j = 0; j < input[b][i].size(); ++j) {
              output[b][i][j] += bias[j];
          }
      }
  }
  return output;  
}

std::vector<std::vector<std::vector<float>>> FeedForwardLayer::backward(const std::vector<std::vector<std::vector<float>>>& input, const std::vector<std::vector<std::vector<float>>>& grad_output) {

