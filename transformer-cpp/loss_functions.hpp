#pragma once
#include <vector>

float cross_entropy_loss(
    const std::vector<std::vector<std::vector<float>>> &output_logits,
    const std::vector<std::vector<int>> &targets);
