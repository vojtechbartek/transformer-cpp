#pragma once
#include <vector>
#include <string>


std::vector<std::vector<float>> load_embeddings(const std::string &filename);
std::vector<int> load_targets(const std::string &filename);
