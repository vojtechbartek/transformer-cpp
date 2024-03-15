#include "self_attention.hpp"
#include "matrix_utils.hpp"
#include <random>

SelfAttention::SelfAttention(int embed_dim, int head_size) : embed_dim_(embed_dim), head_size_(head_size) {
    // Initialize the weights
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<float> distribution(0.0, 0.3); // mean=0, stddev=0.1
    
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


std::vector<std::vector<std::vector<float>>> SelfAttention::forward(const std::vector<std::vector<std::vector<float>>>& input,
                                                       const std::vector<std::vector<std::vector<float>>>& mask) {
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

std::vector<std::vector<std::vector<float>>> SelfAttention::backward(const std::vector<std::vector<std::vector<float>>>& input, const std::vector<std::vector<std::vector<float>>>& grad_output) {
  /*
    * Compute the gradients of the self-attention layer
    * 
    * @param input: the input to the self-attention layer, size (batch_size x seq_len x embed_dim)
    * @param grad_output: the gradient of the loss with respect to the output of the self-attention layer, size (batch_size x seq_len x head_size)
    * @return: the gradient of the loss with respect to the input of the self-attention layer, size (batch_size x seq_len x embed_dim)
  */

  // Compute the gradients with respect to the attention weights
  auto grad_scores = MatrixUtils::rowSoftmaxDerivative(grad_output, softmax_output); // (batch_size x seq_len x seq_len)

  // Compute the gradients with respect to Q, K, and V
  auto grad_Q = MatrixUtils::batchMatrixMultiplication(grad_scores, K); // (batch_size x seq_len x head_size)
  auto grad_K = MatrixUtils::batchMatrixMultiplication(grad_scores, Q); // (batch_size x seq_len x head_size)
  auto grad_V = grad_output; // Direct gradient contribution for V from grad_output, shape (batch_size x seq_len x head_size)

  // Initialize the gradients to zero
  grad_Wq = std::vector<std::vector<float>>(embed_dim_, std::vector<float>(head_size_, 0));
  grad_Wk = std::vector<std::vector<float>>(embed_dim_, std::vector<float>(head_size_, 0));
  grad_Wv = std::vector<std::vector<float>>(embed_dim_, std::vector<float>(head_size_, 0));

  int batch_size = input.size();

  // Accumulate the gradients over the batch
  for (int b = 0; b < batch_size; ++b) {
    auto grad_Wq_b = MatrixUtils::matrixMultiplication(MatrixUtils::matrixTranspose(input[b]), grad_Q[b]); // (embed_dim x head_size)
    auto grad_Wk_b = MatrixUtils::matrixMultiplication(MatrixUtils::matrixTranspose(input[b]), grad_K[b]); // (embed_dim x head_size)
    auto grad_Wv_b = MatrixUtils::matrixMultiplication(MatrixUtils::matrixTranspose(input[b]), grad_V[b]); // (embed_dim x head_size)
    
    // Average the gradients over the batch
    for (int i = 0; i < embed_dim_; ++i) {
      for (int j = 0; j < head_size_; ++j) {
        grad_Wq[i][j] += grad_Wq_b[i][j] / batch_size;
        grad_Wk[i][j] += grad_Wk_b[i][j] / batch_size; 
        grad_Wv[i][j] += grad_Wv_b[i][j] / batch_size;
      }
    }
  }
  // These gradients will be used later to update the weights

  
  auto transpose_Wq = MatrixUtils::matrixTranspose(Wq); // (head_size x embed_dim)
  auto transpose_Wk = MatrixUtils::matrixTranspose(Wk); // (head_size x embed_dim)
  auto transpose_Wv = MatrixUtils::matrixTranspose(Wv); // (head_size x embed_dim)

  auto grad_input_Q = MatrixUtils::batchMatrixMultiplication(grad_Q, transpose_Wq); 
  auto grad_input_K = MatrixUtils::batchMatrixMultiplication(grad_K, transpose_Wk); 
  auto grad_input_V = MatrixUtils::batchMatrixMultiplication(grad_V, transpose_Wv); 
  // Sum up the gradients from Q, K, and V to get the total gradient
  
  std::vector<std::vector<std::vector<float>>> grad_input = MatrixUtils::matrixAddition(grad_input_Q, MatrixUtils::matrixAddition(grad_input_K, grad_input_V)); // (batch_size x seq_len x embed_dim)
  
  return grad_input;
}

std::vector<std::vector<std::vector<float>>> SelfAttention::getSoftmaxOutput() const {
  return softmax_output;
}

std::vector<std::vector<std::vector<float>>> SelfAttention::getWeights() const {
  std::vector<std::vector<std::vector<float>>> weights = {Wq, Wk, Wv};
  return weights;
}

void SelfAttention::update_weights(float learning_rate) {
  // Update the weights using the gradients
  for (int i = 0; i < embed_dim_; i++) {
    for (int j = 0; j < head_size_; j++) {
      Wq[i][j] -= learning_rate * grad_Wq[i][j];
      Wk[i][j] -= learning_rate * grad_Wk[i][j];
      Wv[i][j] -= learning_rate * grad_Wv[i][j];
    }
  }
}
