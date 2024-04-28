#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace MatrixUtils {

template <typename T>
std::vector<std::vector<T>>
matrixAddition(const std::vector<std::vector<T>> &A,
               const std::vector<std::vector<T>> &B) {
  /*
   * Compute the addition of two matrices
   *
   * @param A: a matrix of size (rows x cols)
   * @param B: a matrix of size (rows x cols)
   * @return: the result of the addition of the two matrices, size (rows x cols)
   */
  if (A.size() != B.size() || A[0].size() != B[0].size()) {
    std::cerr << "Matrix A and B are not compatible for addition, A size = "
              << A.size() << "x" << A[0].size() << " and B size = " << B.size()
              << "x" << B[0].size() << std::endl;
    throw std::invalid_argument(
        "Matrix A and B are not compatible for addition");
  }
  int rows = A.size(), cols = A[0].size();
  std::vector<std::vector<T>> result(rows, std::vector<T>(cols, 0));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result[i][j] = A[i][j] + B[i][j];
    }
  }
  return result;
}

template <typename T>
std::vector<std::vector<std::vector<T>>>
matrixAddition(const std::vector<std::vector<std::vector<T>>> &A,
               const std::vector<std::vector<std::vector<T>>> &B) {
  /*
   * Compute the addition of two 3D matrices
   *
   * @param A: a 3D matrix of size (a x b x c)
   * @param B: a 3D matrix of size (a x b x c)
   * @return: the result of the addition of the two 3D matrices, size (a x b x
   * c)
   */
  int a = A.size(), b = A[0].size(), c = A[0][0].size();
  if (A.size() != B.size() || A[0].size() != B[0].size() ||
      A[0][0].size() != B[0][0].size()) {
    std::cerr << "Matrix A and B are not compatible for addition, A size = "
              << A.size() << "x" << A[0].size() << "x" << A[0][0].size()
              << " and B size = " << B.size() << "x" << B[0].size() << "x"
              << B[0][0].size() << std::endl;
    throw std::invalid_argument(
        "Matrix A and B are not compatible for addition");
  }
  std::vector<std::vector<std::vector<T>>> result(
      a, std::vector<std::vector<T>>(b, std::vector<T>(c, 0)));
  for (int i = 0; i < a; ++i) {
    for (int j = 0; j < b; ++j) {
      for (int k = 0; k < c; ++k) {
        result[i][j][k] = A[i][j][k] + B[i][j][k];
      }
    }
  }
  return result;
}

template <typename T>
std::vector<std::vector<T>>
matrixMultiplication(const std::vector<std::vector<T>> &A,
                     const std::vector<std::vector<T>> &B) {
  int a_rows = A.size(), a_cols = A[0].size(), b_rows = B.size(),
      b_cols = B[0].size();
  if (a_cols != b_rows) {
    // matrices are not compatible for multiplication, throw an exception and
    // print the sizes of the matrices
    std::cerr << "Matrix A and B are not compatible for multiplication"
              << std::endl;
    std::cerr << "A : " << A.size() << "x" << A[0].size() << std::endl;
    std::cerr << "B : " << B.size() << "x" << B[0].size() << std::endl;
    throw std::invalid_argument(
        "Matrix A and B are not compatible for multiplication");
  }

  std::vector<std::vector<T>> result(a_rows, std::vector<T>(b_cols, 0));
  for (int i = 0; i < a_rows; ++i) {
    for (int j = 0; j < b_cols; ++j) {
      for (int k = 0; k < a_cols; ++k) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
}

template <typename T>
std::vector<std::vector<std::vector<T>>> batchMatrixMultiplication(
    const std::vector<std::vector<std::vector<T>>> &batch_A,
    const std::vector<std::vector<T>> &B) {
  /*
   * Compute the matrix multiplication of a batch of matrices with a single
   * matrix
   *
   * @param batch_A: a batch of matrices of size (batch x seq_len x embed_dim)
   * @param B: a single matrix of size (embed_dim x head_size)
   * @return: the result of the matrix multiplication of size (batch x seq_len x
   * head_size)
   */
  int batch_size = batch_A.size();
  int seq_len = batch_A[0].size();
  int embed_dim = B.size();
  int head_size = B[0].size();

  std::vector<std::vector<std::vector<T>>> result(
      batch_size,
      std::vector<std::vector<T>>(seq_len, std::vector<T>(head_size, 0)));

  for (int b = 0; b < batch_size; ++b) {
    if (batch_A[b][0].size() != embed_dim) {
      std::cerr
          << "Matrix A and B are not compatible for batched multiplication"
          << std::endl;
      std::cerr << "A : " << batch_A.size() << "x" << batch_A[0].size() << "x"
                << batch_A[0][0].size() << std::endl;
      std::cerr << "B : " << B.size() << "x" << B[0].size() << std::endl;
      throw std::invalid_argument(
          "Matrix A and B are not compatible for multiplication");
    }
    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j < head_size; ++j) {
        for (int k = 0; k < embed_dim; ++k) {
          result[b][i][j] += batch_A[b][i][k] * B[k][j];
        }
      }
    }
  }
  return result;
}

template <typename T>
std::vector<std::vector<std::vector<T>>> batchMatrixMultiplication(
    const std::vector<std::vector<std::vector<T>>> &batch_A,
    const std::vector<std::vector<std::vector<T>>> &batch_B) {
  /*
   * Compute the matrix multiplication of a 3D matrix with another 3D matrix
   *
   * @param batch_A: a batch of matrices of size (a x b x c)
   * @param batch_B: a batch of matrices of size (a x c x d)
   * @return: the result of the matrix multiplication of size (a x b x d)
   */
  int a = batch_A.size();
  int b = batch_A[0].size();
  int c = batch_B[0].size();
  int d = batch_B[0][0].size();

  if (batch_A[0][0].size() != c) {
    // print sizes of both matrices
    std::cerr << "Matrix A and B are not compatible for multiplication"
              << std::endl;
    std::cerr << "A : " << batch_A.size() << "x" << batch_A[0].size() << "x"
              << batch_A[0][0].size() << std::endl;
    std::cerr << "B : " << batch_B.size() << "x" << batch_B[0].size() << "x"
              << batch_B[0][0].size() << std::endl;

    throw std::invalid_argument(
        "Matrix A and B are not compatible for multiplication");
  }
  std::vector<std::vector<std::vector<T>>> result(
      a, std::vector<std::vector<T>>(b, std::vector<T>(d, 0)));
  for (int i = 0; i < a; ++i) {
    for (int j = 0; j < b; ++j) {
      for (int k = 0; k < d; ++k) {
        for (int l = 0; l < c; ++l) {
          result[i][j][k] += batch_A[i][j][l] * batch_B[i][l][k];
        }
      }
    }
  }
  return result;
}

template <typename T>
std::vector<std::vector<T>>
batchMatrixMean(const std::vector<std::vector<std::vector<T>>> &batch_tensor) {
  /*
   * Compute the mean across the batch (first) dimension of a 3D matrix
   *
   * @param batch_tensor: a batch of matrices of size (batch x rows x cols)
   * @return: the mean of the batch of matrices of size (rows x cols)
   */
  int batch_size = batch_tensor.size();
  int rows = batch_tensor[0].size();
  int cols = batch_tensor[0][0].size();

  std::vector<std::vector<T>> mean(rows, std::vector<T>(cols, 0));
  for (const auto &matrix : batch_tensor) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        mean[i][j] += matrix[i][j] / batch_size;
      }
    }
  }
  return mean;
}

template <typename T>
std::vector<T>
batchVectorMean(const std::vector<std::vector<std::vector<T>>> &batch_tensor) {
  /*
   * Compute the mean across the batch (first) and rows (second) dimension of a
   * 3D matrix
   *
   * @param batch_tensor: a batch of matrices of size (batch x rows x cols)
   * @return: the mean of the batch of matrices of size (cols)
   */
  int batch_size = batch_tensor.size();
  int cols = batch_tensor[0][0].size();

  std::vector<T> mean(cols, 0);

  for (const auto &matrix : batch_tensor) {
    for (const auto &row : matrix) {
      for (int j = 0; j < cols; ++j) {
        mean[j] += row[j] / (batch_size * matrix.size());
      }
    }
  }
  return mean;
}

template <typename T>
std::vector<std::vector<T>>
matrixTranspose(const std::vector<std::vector<T>> &A) {
  int rows = A.size(), cols = A[0].size();
  // initialize the result matrix with the size of the transposed matrix
  std::vector<std::vector<T>> result(cols, std::vector<T>(rows, T(0)));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result[j][i] = A[i][j];
    }
  }

  return result;
}

template <typename T>
std::vector<std::vector<std::vector<T>>>
batchMatrixTranspose(const std::vector<std::vector<std::vector<T>>> &batch_A) {
  /*
   * Compute the transpose of a batch of matrices
   *
   * @param batch_A: a batch of matrices of size (batch x rows x cols)
   * @return: the transpose of the batch of matrices of size (batch x cols x
   * rows)
   */
  int batch_size = batch_A.size();
  int rows = batch_A[0].size();
  int cols = batch_A[0][0].size();
  std::vector<std::vector<std::vector<T>>> result(
      batch_size, std::vector<std::vector<T>>(cols, std::vector<T>(rows, 0)));
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        result[b][j][i] = batch_A[b][i][j];
      }
    }
  }
  return result;
}

template <typename T>
std::vector<std::vector<T>> rowSoftmax(const std::vector<std::vector<T>> &A) {
  /*
   * Compute the softmax of a matrix
   *
   * @param A: a matrix of size (seq_len x embed_dim)
   * @return: the softmax of the matrix
   */
  int seq_len = A.size(), embed_dim = A[0].size();
  std::vector<std::vector<T>> softmax_scores(seq_len,
                                             std::vector<T>(embed_dim, 0));
  for (int i = 0; i < seq_len; ++i) {
    T max = *std::max_element(A[i].begin(), A[i].end()); // Find max element
    // T sum = 1e-10; // to avoid division by zero
    T sum = 0.0;
    for (int j = 0; j < embed_dim; ++j) {
      // Subtract max from each element to avoid overflow
      softmax_scores[i][j] = std::exp(A[i][j] - max);
      sum += softmax_scores[i][j];
    }

    // divide each element by the sum of all elements
    for (int j = 0; j < embed_dim; ++j) {
      softmax_scores[i][j] /= sum;
    }
  }

  return softmax_scores;
}

template <typename T>
std::vector<std::vector<std::vector<T>>>
rowSoftmax(const std::vector<std::vector<std::vector<T>>> &A) {
  /*
   * Compute the softmax of a batch of matrices
   *
   * @param A: a batch of matrices of size (batch x seq_len x embed_dim)
   * @return: the softmax of the batch of matrices
   */
  int batch_size = A.size();
  std::vector<std::vector<std::vector<T>>> softmax_scores(batch_size);
  for (int b = 0; b < batch_size; ++b) {
    softmax_scores[b] = rowSoftmax(A[b]);
  }
  return softmax_scores;
}

template <typename T>
std::vector<std::vector<T>>
rowSoftmaxDerivative(const std::vector<std::vector<T>> &grad_output,
                     const std::vector<std::vector<T>> &softmax_output) {
  /*
   * Compute the derivative of the softmax function with respect to the input
   *
   * @param grad_output: the gradient of the loss with respect to the output of
   * the softmax function, shape (seq_len x embed_dim)
   * @param softmax_output: the output of the softmax function, shape (seq_len x
   * embed_dim)
   * @return: the gradient of the loss with respect to the input of the softmax
   * function, shape (seq_len x embed_dim)
   */
  size_t seq_len = softmax_output.size();
  size_t embed_dim = softmax_output[0].size();

  std::vector<std::vector<T>> grad_softmax(seq_len,
                                           std::vector<T>(embed_dim, 0.0));

  for (size_t i = 0; i < seq_len; i++) {
    for (size_t j = 0; j < embed_dim; j++) {
      T diagonal_term = softmax_output[i][j] * (1 - softmax_output[i][j]);
      T off_diagonal_term = -softmax_output[i][j];

      T gradient = 0.0;
      for (size_t k = 0; k < embed_dim; k++) {
        gradient +=
            grad_output[i][k] *
            (k == j ? diagonal_term : off_diagonal_term * softmax_output[i][k]);
      }
      grad_softmax[i][j] = gradient;
    }
  }
  return grad_softmax;
}

template <typename T>
std::vector<std::vector<std::vector<T>>> rowSoftmaxDerivative(
    const std::vector<std::vector<std::vector<T>>> &grad_output,
    const std::vector<std::vector<std::vector<T>>> &softmax_output) {
  /*
   * Compute the derivative of the softmax function with respect to the input
   * for a batch of matrices
   *
   * @param grad_output: the gradient of the loss with respect to the output of
   * the softmax function, shape (batch x seq_len x embed_dim)
   * @param softmax_output: the output of the softmax function, shape (batch x
   * seq_len x embed_dim)
   * @return: the gradient of the loss with respect to the input of the softmax
   * function, shape (batch x seq_len x embed_dim)
   */

  size_t batch_size = softmax_output.size();
  std::vector<std::vector<std::vector<T>>> grad_softmax(batch_size);
  for (size_t b = 0; b < batch_size; b++) {
    grad_softmax[b] = rowSoftmaxDerivative(grad_output[b], softmax_output[b]);
  }
  return grad_softmax;
}

} // namespace MatrixUtils
