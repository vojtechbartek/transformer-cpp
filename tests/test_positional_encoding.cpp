#include "../transformer-cpp/positional_encoding.hpp"
#include <cmath>
#include <iostream>

int LENGTH = 10;
int EMBEDDING_DIM = 512;

bool floatAlmostEqual(float a, float b, float epsilon = 0.0001) {
	return std::fabs(a - b) < epsilon;
}

bool checkPositionalEncodingProperties(std::vector<std::vector<float>> &positionalEncoding, int length, int embeddingDim) {
	if (positionalEncoding.size() != length) {
		std::cerr << "Error: positionalEncoding.size() != length" << std::endl;
		return false;
	}
	for (const auto &row: positionalEncoding) {
		if (row.size() != embeddingDim) {
			std::cerr << "Error: row.size() != embeddingDim" << std::endl;
			return false;
		}
		for (const auto &value: row) {
			if (value < -1.0 || value > 1.0) {
				std::cerr << "Error: value < -1.0 || value > 1.0" << std::endl;
				return false;
			}
		}
	}

	return true;
}

bool checkPositionalEncodingValue(std::vector<std::vector<float>> &positionalEncoding, int length, int embeddingDim) {
	// check specific value
	// 0th row
	// 0th column
	if (positionalEncoding[0][0] != 0.0) {
		std::cerr << "Error: positionalEncoding[0][0] = " << positionalEncoding[0][0] << " != 0.0" << std::endl;
		return false;
	}

	// 0st row
	// 1st column
	// cos (0 / 10000^((2*1)/512)) = cos(0) = 1.0
	if (positionalEncoding[0][1] != 1.0) {
		std::cerr << "Error: positionalEncoding[0][1] = " << positionalEncoding[0][1] << " != 1.0" << std::endl;
		return false;
	}

	// 2nd row
	// 2nd column
	// sin (2 / 10000^((2*2)/512)) 
	if (!floatAlmostEqual(positionalEncoding[2][2], std::sin(2 / std::pow(10000, 4.0/512)))) {
		std::cerr << "Error: positionalEncoding[2][2] = " << positionalEncoding[2][2] << " != sin(2 / 10000^((2*2)/" << embeddingDim << "))" << " = " << std::sin(2 / std::pow(10000, 4.0/512)) << std::endl;
		return false;
	}

	return true;
}

int main() {
	auto positionalEncoding = PositionalEncoding::generate(LENGTH, EMBEDDING_DIM);
	if (checkPositionalEncodingProperties(positionalEncoding, LENGTH, EMBEDDING_DIM)) {
		std::cout << "[+] Encoding properties Test passed" << std::endl;
	} else {
		std::cout << "[-] Test failed" << std::endl;
	}

	if (checkPositionalEncodingValue(positionalEncoding, LENGTH, EMBEDDING_DIM)) {
		std::cout << "[+] Encoding value Test passed" << std::endl;
	} else {
		std::cout << "[-] Test failed" << std::endl;
	}	

	return 0;
}
