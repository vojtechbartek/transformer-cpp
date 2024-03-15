#pragma once
#include <vector>

std::vector<std::vector<float>> load_embeddings(const std::string &filename);
std::vector<int> load_targets(const std::string &filename);
