#pragma once
#include <vector>
#include <cmath>


namespace PositionalEncoding {

	static std::vector<std::vector<float>> generate(int length, int embedding_dim) {
	
	// Initialize the position encoding matrix
	std::vector<std::vector<float>> position_encoding(length, std::vector<float>(embedding_dim));
	
	for (int pos = 0; pos < length; ++pos) {
		for (int i = 0; i < embedding_dim; ++i) {
		float omega = 1.0 / pow(10000, (2 * i) / static_cast<float>(embedding_dim));
		position_encoding[pos][i] = (i % 2 == 0) ? std::sin(pos * omega) : std::cos(pos * omega);
		}
	}
	
	return position_encoding;
	}	
};

