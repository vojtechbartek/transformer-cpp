#include "self_attention.hpp"
#include "matrix_utils.hpp"
#include <random>

SelfAttention::SelfAttention(int embed_dim, int num_heads) : embed_dim(embed_dim), num_heads(num_heads) {
    // Initialize the weights
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<float> distribution(0.0, 0.1); // mean=0, stddev=0.1
    
    Wq = std::vector<std::vector<float>>(embed_dim, std::vector<float>(embed_dim));
    Wk = std::vector<std::vector<float>>(embed_dim, std::vector<float>(embed_dim));
    Wv = std::vector<std::vector<float>>(embed_dim, std::vector<float>(embed_dim));

    for (int i = 0; i < embed_dim; i++) {
        for (int j = 0; j < embed_dim; j++) {
            Wq[i][j] = distribution(generator);
            Wk[i][j] = distribution(generator);
            Wv[i][j] = distribution(generator);
        }
    }
}


std::vector<std::vector<float>> SelfAttention::forward(std::vector<std::vector<float>>& input,
                                                       std::vector<std::vector<float>>& mask) {
  Q = MatrixUtils::matrixMultiplication(input, Wq);
  K = MatrixUtils::matrixMultiplication(input, Wk);
  V = MatrixUtils::matrixMultiplication(input, Wv);


  auto attention = MatrixUtils::matrixMultiplication(Q, MatrixUtils::matrixTranspose(K));

  // Scale the attention scores
  float scale = std::sqrt(static_cast<float>(embed_dim_));
  for (auto &row : attention) {
    for (auto &val : row) {
      val /= scale;
    }
  }

  // Apply the mask
  if (!mask.empty()) {
    for (int i = 0; i < attention.size(); i++) {
      for (int j = 0; j < attention[0].size(); j++) {
        if (mask[i][j] == 0) {
          attention[i][j] = -1e9; // Set to a large negative value
        }

  // Softmax
  auto attention_weights = MatrixUtils::rowSoftmax(attention);

  // Multiply the attention weights with the values
  auto output = MatrixUtils::matrixMultiplication(attention_weights, V);

  return output;
}

std::vector<std::vector<float>> SelfAttention::backward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& grad_output) {
  // Compute the gradients with respect to the attention weights
  auto grad_scores = MatrixUtils::rowSoftmaxDerivative(grad_output, softmax_output);

  // Compute the gradients with respect to Q, K, and V
  auto grad_Q = MatrixUtils::matrixMultiplication(grad_scores, MatrixUtils::matrixTranspose(K));
  auto grad_K = MatrixUtils::matrixMultiplication(MatrixUtils::matrixTranspose(grad_scores), Q);
  auto grad_V = grad_output; // Direct gradient contribution for V from grad_output
  
  // Compute the gradients with respect to Wq, Wk, and Wv
  grad_Wq = MatrixUtils::matrixMultiplication(MatrixUtils::matrixTranspose(input), grad_Q);
  grad_Wk = MatrixUtils::matrixMultiplication(MatrixUtils::matrixTranspose(input), grad_K);
  grad_Wv = MatrixUtils::matrixMultiplication(MatrixUtils::matrixTranspose(input), grad_V);

  // Compute the gradients with respect to the input
  auto grad_input = MatrixUtils::matrixMultiplication(grad_Q, MatrixUtils::matrixTranspose(Wq)) +
                    MatrixUtils::matrixMultiplication(grad_K, MatrixUtils::matrixTranspose(Wk)) +
                    MatrixUtils::matrixMultiplication(grad_V, MatrixUtils::matrixTranspose(Wv));

  return grad_input;
}
