#include <cmath>
#include <vector>


float cross_entropy_loss(const std::vector<std::vector<std::vector<float>>>& output_logits, const std::vector<std::vector<int>>& targets) {
    float loss = 0.0;
    int batch_size = output_logits.size();
    int seq_len = output_logits[0].size();

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            int target_idx = targets[i][j];
            loss -= std::log(output_logits[i][j][target_idx]);
        }
    }

    loss /= (batch_size * seq_len);
    return loss;
}
