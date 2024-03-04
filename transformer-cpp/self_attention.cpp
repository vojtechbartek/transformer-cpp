#include "self_attention.hpp"
#include "matrix_utils.hpp"
#include <random>

SelfAttention::SelfAttention(int embed_dim, int head_size) : embed_dim_(embed_dim), head_size_(head_size) {
    // Initialize the weights
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<float> distribution(0.0, 0.1); // mean=0, stddev=0.1
    
    Wq = std::vector<std::vector<float>>(embed_dim, std::vector<float>(head_size));
    Wk = std::vector<std::vector<float>>(embed_dim, std::vector<float>(head_size));
    Wv = std::vector<std::vector<float>>(embed_dim, std::vector<float>(head_size));

    for (int i = 0; i < embed_dim; i++) {
        for (int j = 0; j < head_size; j++) {
            Wq[i][j] = distribution(generator);
            Wk[i][j] = distribution(generator);
            Wv[i][j] = distribution(generator);
        }
    }
}


std::vector<std::vector<std::vector<float>>> SelfAttention::forward(std::vector<std::vector<std::vector<float>>>& input,
                                                       std::vector<std::vector<std::vector<float>>>& mask) {
  /*
   * Compute the forward pass of the self-attention layer
   *
   * @param input: the input to the self-attention layer (batch_size x seq_len x embed_dim)
   * @param mask: the mask to apply to the attention scores (batch_size x seq_len x seq_len)
   * @return: the output of the self-attention layer (batch_size x seq_len x head_size)
   *
   * Note: The mask is a tensor of 0s and 1s, where 0s indicate that the attention score should be ignored
   */


  Q = MatrixUtils::batchMatrixMultiplication(input, Wq); // (batch_size x seq_len x head_size)
  K = MatrixUtils::batchMatrixMultiplication(input, Wk); // (batch_size x seq_len x head_size)
  V = MatrixUtils::batchMatrixMultiplication(input, Wv); // (batch_size x seq_len x head_size)


  auto batch_attention = MatrixUtils::batchMatrixMultiplication(Q, MatrixUtils::batchMatrixTranspose(K)); // (batch_size x seq_len x seq_len)
  // batch_attention tells us how much each token influences the other tokens in the sequence

  // Scale the attention scores
  float scale = std::sqrt(static_cast<float>(embed_dim_));
  for (int b = 0; b < batch_attention.size(); ++b) {
    for (int i = 0; i < batch_attention[0].size(); ++i) {
      for (int j = 0; j < batch_attention[0][0].size(); ++j) {
        // Apply the mask and scale where mask is 1
        batch_attention[b][i][j] = mask[b][i][j] == 0 ? -1e9 : batch_attention[b][i][j] / scale;
      }
    }
  } 


  // Softmax
  auto attention_weights = MatrixUtils::rowSoftmax(batch_attention); // (batch_size x seq_len x seq_len)
  
  // Save the softmax output for the backward pass
  softmax_output = attention_weights;

  // Multiply the attention weights with the values
  auto output = MatrixUtils::batchMatrixMultiplication(attention_weights, V); // (batch_size x seq_len x head_size)

  return output;
  }

std::vector<std::vector<float>> SelfAttention::backward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& grad_output) {
  /*
    * Compute the gradients of the self-attention layer
    * 
    * @param input: the input to the self-attention layer
    * @param grad_output: the gradient of the loss with respect to the output of the self-attention layer
    * @return: the gradient of the loss with respect to the input of the self-attention layer
  */

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
  auto grad_input = MatrixUtils::matrixAddition(MatrixUtils::matrixMultiplication(grad_Q, MatrixUtils::matrixTranspose(Wq)),
                                               MatrixUtils::matrixAddition(MatrixUtils::matrixMultiplication(grad_K, MatrixUtils::matrixTranspose(Wk)),
                                                                           MatrixUtils::matrixMultiplication(grad_V, MatrixUtils::matrixTranspose(Wv))));

  return grad_input;
}
