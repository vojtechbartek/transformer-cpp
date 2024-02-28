#include "../transformer-cpp/activation_functions.hpp"
#include <iostream>
#include <cmath>

// Helper function to compare two arrays with a given tolerance
bool compareArrays(float* arr1, float* arr2, int size, float tolerance = 1e-5) {
	for (int i = 0; i < size; i++) {
		if (std::abs(arr1[i] - arr2[i]) > tolerance) {
			return false;
		}
	}
	return true;
}

int main() {
	bool allTestsPassed = true;

	// Test the ReLU activation function
	float reluInput[] = {1.0f, 0.0f, -1.0f, 2.0f};
	float reluExpected[] = {1.0f, 0.0f, 0.0f, 2.0f};
	float reluOutput[4];
	ActivationFunction::relu(reluInput, reluOutput, 4);
	if (!compareArrays(reluOutput, reluExpected, 4)) {
		std::cout << "[-] ReLU test failed" << std::endl;
		allTestsPassed = false;
	} else {
		std::cout << "[+] ReLU test passed" << std::endl;
	}

	// Test the ReLU derivative function
	float reluDerivativeInput[] = {1.0f, 0.0f, -1.0f, 2.0f};
	float reluDerivativeExpected[] = {1.0f, 0.0f, 0.0f, 1.0f};
	float reluDerivativeOutput[4];
	ActivationFunction::reluDerivative(reluDerivativeInput, reluDerivativeOutput, 4);
	if (!compareArrays(reluDerivativeOutput, reluDerivativeExpected, 4)) {
		std::cout << "[-] ReLU derivative test failed" << std::endl;
		allTestsPassed = false;
	} else {
		std::cout << "[+] ReLU derivative test passed" << std::endl;
	}

	// Test the softmax function
 	float softmaxInput[] = {1.4f, 2.0f, -3.0f, 4.02f};
	float softmaxExpected[] = {
		exp(1.4f) / (exp(1.4f) + exp(2.0f) + exp(-3.0f) + exp(4.02f)),
		exp(2.0f) / (exp(1.4f) + exp(2.0f) + exp(-3.0f) + exp(4.02f)),
		exp(-3.0f) / (exp(1.4f) + exp(2.0f) + exp(-3.0f) + exp(4.02f)),
		exp(4.02f) / (exp(1.4f) + exp(2.0f) + exp(-3.0f) + exp(4.02f))
	};
	float softmaxOutput[4];
	ActivationFunction::softmax(softmaxInput, softmaxOutput, 4);
	if (!compareArrays(softmaxOutput, softmaxExpected, 4)) {
		std::cout << "[-] Softmax test failed" << std::endl;
		allTestsPassed = false;
	} else {
		std::cout << "[+] Softmax test passed" << std::endl;
	}

	
	return allTestsPassed ? 0 : 1;
}

