#include "feed_forward_layer.hpp"
#include "matrix_utils.hpp"
#include "activation_functions.hpp"
#include <vector>
#include <random>

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
  hidden_relu = ActivationFunctions::ReLU(hidden); // batch_size x seq_len x hidden_dim
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
    /*
     * input: batch_size x seq_len x head_size 
     * grad_output: batch_size x seq_len x output_dim
     *
     * output: batch_size x seq_len x head_size
    */
    // gradient through the second linear transformation
    std::vector<std::vector<std::vector<float>>> grad_hidden_relu = MatrixUtils::batchMatrixMultiplication(grad_output, MatrixUtils::matrixTranspose(W2)); // batch_size x seq_len x hidden_dim
    // Mask the gradient based on the relu output
    std::vector<std::vector<std::vector<float>>> grad_hidden = ActivationFunctions::ReLU_derivative(hidden_relu, grad_hidden_relu); // batch_size x seq_len x hidden_dim
    
    // gradient through the first linear transformation
    std::vector<std::vector<std::vector<float>>> grad_input = MatrixUtils::batchMatrixMultiplication(grad_hidden, MatrixUtils::matrixTranspose(W1)); // batch_size x seq_len x input_dim

    // compute gradients w.r.t. weights and biases
    std::vector<std::vector<std::vector<float>>> batch_grad_W2 = MatrixUtils::batchMatrixMultiplication(MatrixUtils::batchMatrixTranspose(hidden_relu), grad_output); // batch_size x hidden_dim x output_dim
    std::vector<std::vector<std::vector<float>>> batch_grad_W1 = MatrixUtils::batchMatrixMultiplication(MatrixUtils::batchMatrixTranspose(input), grad_hidden); // batch_size x input_dim x hidden_dim
    
    // average over the batch
    grad_W2 = MatrixUtils::batchMatrixMean(batch_grad_W2); // hidden_dim x output_dim
    grad_W1 = MatrixUtils::batchMatrixMean(batch_grad_W1); // input_dim x hidden_dim

    grad_b2 = MatrixUtils::batchVectorMean(grad_output); // output_dim
    grad_b1 = MatrixUtils::batchVectorMean(grad_hidden); // hidden_dim

    // return the gradient w.r.t. the input
    return grad_input; // batch_size x seq_len x input_dim
} 

std::vector<std::vector<std::vector<float>>> FeedForwardLayer::get_grad_weights() {
    std::vector<std::vector<std::vector<float>>> grad_weights = {grad_W1, grad_W2};
    return grad_weights;
}

std::vector<std::vector<float>> FeedForwardLayer::get_grad_biases() {
    std::vector<std::vector<float>> grad_biases = {grad_b1, grad_b2};
    return grad_biases;
}

void FeedForwardLayer::update_weights(float learning_rate) {
    for (int i = 0; i < input_dim; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            W1[i][j] -= learning_rate * grad_W1[i][j];
        }
    }

    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < output_dim; j++) {
            W2[i][j] -= learning_rate * grad_W2[i][j];
        }
    }

    for (int i = 0; i < hidden_dim; i++) {
        b1[i] -= learning_rate * grad_b1[i];
    }

    for (int i = 0; i < output_dim; i++) {
        b2[i] -= learning_rate * grad_b2[i];
    }
}
