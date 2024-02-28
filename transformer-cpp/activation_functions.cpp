#include "activation_functions.hpp"
#include <algorithm>
#include <cmath>


void ActivationFunction::relu(float* z, float* output, int size) {
  for (int i = 0; i < size; i++) {
    output[i] = std::max(0.0f, z[i]);
  }
} 

void ActivationFunction::reluDerivative(float* z, float* output, int size) {
  // Derivative of ReLU is 1 if z > 0, 0 otherwise
  for (int i = 0; i < size; i++) {
    output[i] = z[i] > 0 ? 1.0f : 0.0f;
  }
}

void ActivationFunction::softmax(float* z, float* output, int size) {
  float max = *std::max_element(z, z + size); // Find max element
  float sum = 0.0f;

  // Subtract max from each element to avoid overflow
  for (int i = 0; i < size; i++) {
    output[i] = std::exp(z[i] - max); // numerical stability
    sum += output[i];
  }
 
  // divide each element by the sum of all elements
  for (int i = 0; i < size; i++) {
    output[i] /= sum;
  }
}

