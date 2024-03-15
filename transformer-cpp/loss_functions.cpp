#include <cmath>
#include <vector>
#include <cassert>


float cross_entropy_loss(const std::vector<std::vector<std::vector<float>>>& output_logits, const std::vector<std::vector<int>>& targets) {
    float loss = 0.0;
    int batch_size = output_logits.size();
    int seq_len = output_logits[0].size();

    assert(batch_size == targets.size());
    assert(seq_len == targets[0].size());

    float eps = 1e-8;

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            int target_idx = targets[i][j];
            assert(target_idx >= 0 && target_idx < output_logits[i][j].size());
            loss -= std::log(std::max(output_logits[i][j][target_idx], eps));
            // loss -= std::log(output_logits[i][j][target_idx]);
        }
    }

    loss /= (batch_size * seq_len);
    return loss;
}
