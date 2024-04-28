#include "../transformer-cpp/activation_functions.hpp"
#include <cmath>
#include <iostream>

// Helper function to compare two matrices with a given tolerance
bool compare_matrices(const std::vector<std::vector<std::vector<float>>> &a,
                      const std::vector<std::vector<std::vector<float>>> &b,
                      float eps = 1e-6) {
  int batch_size = a.size();
  int seq_len = a[0].size();
  int n = a[0][0].size();

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < seq_len; j++) {
      for (int k = 0; k < n; k++) {
        if (std::abs(a[i][j][k] - b[i][j][k]) > eps) {
          return false;
        }
      }
    }
  }

  return true;
}

int main() {

  int batch_size = 2;
  int seq_len = 3;
  int n = 4;

  // Test the ReLU activation function
  std::vector<std::vector<std::vector<float>>> input = {
      {{1.3f, -2.0f, 0.0f, 1.0f},
       {0.0f, 1.0f, -1.12f, 5.0f},
       {0.0f, 0.0f, 0.31f, -0.1f}},
      {{-1.3f, 0.0f, 0.0f, -3.0f},
       {1.1f, 21.2f, -1.0f, 0.0f},
       {0.2f, 0.0f, -0.31f, 0.1f}}};

  std::vector<std::vector<std::vector<float>>> expected = {
      {{1.3f, 0.0f, 0.0f, 1.0f},
       {0.0f, 1.0f, 0.0f, 5.0f},
       {0.0f, 0.0f, 0.31f, 0.0f}},
      {{0.0f, 0.0f, 0.0f, 0.0f},
       {1.1f, 21.2f, 0.0f, 0.0f},
       {0.2f, 0.0f, 0.0f, 0.1f}}};

  std::vector<std::vector<std::vector<float>>> output =
      ActivationFunctions::ReLU(input);
  if (!compare_matrices(output, expected)) {
    std::cout << "[-] ReLU test failed" << std::endl;
  } else {
    std::cout << "[+] ReLU test passed" << std::endl;
  }

  // Test the ReLU derivative function
  input = {{{1.3f, -2.0f, 0.0f, 1.0f},
            {0.0f, 1.0f, -1.12f, 5.0f},
            {0.0f, 0.0f, 0.31f, -0.1f}},
           {{-1.3f, 0.0f, 0.0f, -3.0f},
            {1.1f, 21.2f, -1.0f, 0.0f},
            {0.2f, 0.0f, -0.31f, 0.1f}}};
  std::vector<std::vector<std::vector<float>>> gradient = {
      {{4.0f, -1.0f, 2.0f, -1.2f},
       {12.0f, 1.123f, 1.1f, 3.0f},
       {0.1f, 1, 2, -2}},
      {{2.0f, 1.0f, 1.0f, -1.53f},
       {2.0f, 1.1f, 1.1f, -3.0f},
       {0.1f, -1.0f, 2, 2}},
  };

  expected = {{{4.0f, 0.0f, 0.0f, -1.2f},
               {0.0f, 1.123f, 0.0f, 3.0f},
               {0.0f, 0.0f, 2.0f, 0.0f}},
              {{0.0f, 0.0f, 0.0f, 0.0f},
               {2.0f, 1.1f, 0.0f, 0.0f},
               {0.1f, 0.0f, 0.0f, 2.0f}}};

  std::vector<std::vector<std::vector<float>>> output_derivative =
      ActivationFunctions::ReLU_derivative(input, gradient);
  if (!compare_matrices(output_derivative, expected)) {
    std::cout << "[-] ReLU derivative test failed" << std::endl;
  } else {
    std::cout << "[+] ReLU derivative test passed" << std::endl;
  }

  return 0;
}
