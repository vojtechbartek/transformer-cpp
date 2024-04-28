#pragma once

#include <algorithm>
#include <vector>

namespace ActivationFunctions {
// ReLU activation function
template <typename T>
std::vector<std::vector<std::vector<T>>>
ReLU(const std::vector<std::vector<std::vector<T>>> &input) {
  /*
   * ReLU activation function for a 3D tensor of shape (batch_size, seq_len, n)
   * input: 3D tensor of shape (batch_size, seq_len, n)
   *
   * output: 3D tensor of shape (batch_size, seq_len, n)
   */

  int batch_size = input.size();
  int seq_len = input[0].size();
  int n = input[0][0].size();

  std::vector<std::vector<std::vector<T>>> output(
      batch_size, std::vector<std::vector<T>>(seq_len, std::vector<T>(n, 0)));

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < seq_len; j++) {
      for (int k = 0; k < n; k++) {
        output[i][j][k] = std::max(0.0f, input[i][j][k]);
      }
    }
  }

  return output;
}

template <typename T>
std::vector<std::vector<std::vector<T>>>
ReLU_derivative(const std::vector<std::vector<std::vector<T>>> &input) {
  /*
   * ReLU derivative activation function for a 3D tensor of shape (batch_size,
   * seq_len, n) input: 3D tensor of shape (batch_size, seq_len, n)
   *
   * output: 3D tensor of shape (batch_size, seq_len, n)
   */

  int batch_size = input.size();
  int seq_len = input[0].size();
  int n = input[0][0].size();

  std::vector<std::vector<std::vector<T>>> output(
      batch_size, std::vector<std::vector<T>>(seq_len, std::vector<T>(n, 0.0)));

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < seq_len; j++) {
      for (int k = 0; k < n; k++) {
        output[i][j][k] = input[i][j][k] > 0 ? 1 : 0;
      }
    }
  }

  return output;
}

template <typename T>
std::vector<std::vector<std::vector<T>>>
ReLU_derivative(const std::vector<std::vector<std::vector<T>>> &input,
                const std::vector<std::vector<std::vector<T>>> &gradient) {
  /*
   * ReLU derivative activation function for a 3D tensor of shape (batch_size,
   * seq_len, n), it also takes the gradient of the loss with respect to the
   * output of the ReLU layer and masks the gradient where the ReLU layer output
   * is smaller than 0 (ReLU derivative is 0 where the ReLU output is smaller
   * than 0)
   *
   * input: output of the ReLU layer, 3D tensor of shape (batch_size, seq_len,
   * n) gradient: gradient of the loss with respect to the output of the ReLU
   * layer, 3D tensor of shape (batch_size, seq_len, n)
   *
   * output: masked 3D tensor of shape (batch_size, seq_len, n)
   */

  int batch_size = input.size();
  int seq_len = input[0].size();
  int n = input[0][0].size();

  std::vector<std::vector<std::vector<T>>> output = gradient;

  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j < n; ++j) {
        output[b][i][j] = input[b][i][j] > 0 ? output[b][i][j] : 0;
      }
    }
  }

  return output;
}

}; // namespace ActivationFunctions
