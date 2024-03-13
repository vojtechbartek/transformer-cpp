#include "transformer_model.hpp"
#include "loss_functions.hpp"
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <vector>



int main() {
    // Read YAML configuration file
    YAML::Node config = YAML::LoadFile("configs/config.yaml");

    // Create a transformer model
    Transformer model(config);

    int batch_size = config["batch_size"].as<int>();
    int seq_len = config["seq_len"].as<int>();
    int vocab_size = config["vocab_size"].as<int>();
    int embedding_dim = config["embedding_dim"].as<int>();

    // Load input data in csv format
    std::vector<std::vector<std::vector<float>>> input_data(batch_size, std::vector<std::vector<float>>(seq_len, std::vector<float>(embedding_dim, 0.8))); 
    std::vector<std::vector<int>> target_data(batch_size, std::vector<int>(seq_len, 2));

    int num_epochs = config["num_epochs"].as<int>();
    float learning_rate = config["learning_rate"].as<float>();
    float loss;


    // Train the model
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Epoch " << epoch +1 << std::endl;

        // Forward pass
        std::vector<std::vector<std::vector<float>>> output_logits = model.forward(input_data);

        // Compute loss
        loss = cross_entropy_loss(output_logits, target_data);
        std::cout << "  Loss: " << loss << std::endl;
        
        // Backward pass
        std::vector<std::vector<std::vector<float>>> grad_output(output_logits.size(), std::vector<std::vector<float>>(output_logits[0].size(), std::vector<float>(output_logits[0][0].size(), 0.0)));

        // Compute gradients
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < output_logits[0].size(); j++) {
                grad_output[i][j][target_data[i][j]] = -1.0 / output_logits[i][j][target_data[i][j]];
            }
        }
        std::vector<std::vector<std::vector<float>>> grad_input = model.backward(grad_output, output_logits);

        // Update weights
        model.update_weights(learning_rate);
    }

    // TODO Save the model
    return 0;
}

