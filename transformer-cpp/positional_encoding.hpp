#pragma once
#include <vector>

class PositionalEncoding {
public:
	static std::vector<std::vector<float>> generate (int length, int embeddingDim);
};

