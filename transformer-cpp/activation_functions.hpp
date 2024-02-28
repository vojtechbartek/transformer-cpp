#pragma once

class ActivationFunction {
public:
	// ReLU activation function
	static void relu(float*z, float* output, int size);

	// Derivative of ReLU activation function
	static void reluDerivative(float* z, float* output, int size);

	// Softmax activation function
	static void softmax(float* z, float* output, int size);

};

