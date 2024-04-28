#include "load_csv.hpp"
#include "loss_functions.hpp"
#include "transformer_model.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

int main() {
  // Read YAML configuration file
  YAML::Node config = YAML::LoadFile("configs/config.yaml");

  // Create a transformer model
  Transformer model(config);

  int batch_size = config["batch_size"].as<int>();
  int seq_len = config["seq_len"].as<int>();
  int vocab_size = config["vocab_size"].as<int>();
  int embedding_dim = config["embedding_dim"].as<int>();
  std::string input_data_file = config["input_data_file"].as<std::string>();
  std::string target_data_file = config["target_data_file"].as<std::string>();
  int num_epochs = config["num_epochs"].as<int>();
  float learning_rate = config["learning_rate"].as<float>();

  // Load input data in csv format
  std::vector<std::vector<float>> input_data = load_embeddings(input_data_file);
  std::vector<int> target_data = load_targets(target_data_file);

  if (input_data.size() != target_data.size()) {
    std::cerr << "Input and target data have different sizes" << std::endl;
    std::cerr << "Input data size: " << input_data.size() << std::endl;
    std::cerr << "Target data size: " << target_data.size() << std::endl;
    return 1;
  }

  if (input_data[0].size() != embedding_dim) {
    std::cerr << "Input data has wrong dimension" << std::endl;
    std::cerr << "Expected: " << embedding_dim << std::endl;
    std::cerr << "Got: " << input_data[0].size() << std::endl;
    return 1;
  }

  // Convert input data to batched format for training
  // input has shape (batch_size, seq_len, embedding_dim)
  // where batch_size is the number of samples in the batch
  // seq_len is the length of the sequence, meaning the number of whole vectors
  // and embedding_dim is the dimension of the vectors
  std::vector<std::vector<std::vector<float>>> input_data_batched(
      batch_size, std::vector<std::vector<float>>(
                      seq_len, std::vector<float>(embedding_dim, 0.0)));
  std::vector<std::vector<int>> target_data_batched(
      batch_size, std::vector<int>(seq_len, 0));

  // Train the model
  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    std::cout << "Epoch " << epoch + 1 << std::endl;
    auto start = std::chrono::steady_clock::now();

    // fill the batched input and target data, epoch ends when all data is used
    // once if the data is not divisible by the batch size, the last batch will
    // be smaller At each step in the epoch, we will load batch_size * seq_len
    // samples from the input data
    int step = 0;
    while (step < (input_data.size() - batch_size * seq_len)) {
      for (int batch = 0; batch < batch_size; batch++) {
        for (int seq = 0; seq < seq_len; seq++) {
          input_data_batched[batch][seq] = input_data[step];
          target_data_batched[batch][seq] = target_data[step];
          step++;
        }
      }

      // Forward pass
      std::vector<std::vector<std::vector<float>>> output_logits =
          model.forward(input_data_batched);
      bool first = true;
      for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
          int max_index = 0;
          float max_value = -1.0;
          for (int v = 0; v < vocab_size; v++) {
            if (output_logits[b][s][v] > max_value) {
              max_value = output_logits[b][s][v];
              max_index = v;
            }
          }
          if (first) {
            std::cout << "    Predicted: " << max_index << " "
                      << " with probability " << max_value << std::endl;
            std::cout << "    Target: " << target_data_batched[b][s]
                      << std::endl;
            std::cout << "    Target predicted probability: "
                      << output_logits[b][s][target_data_batched[b][s]]
                      << std::endl;
            first = false;
          }
        }
      }

      // Compute loss
      float loss = cross_entropy_loss(output_logits, target_data_batched);
      std::cout << "    Loss: " << loss << std::endl;

      // Backward pass
      std::vector<std::vector<std::vector<float>>> grad_output(
          output_logits.size(),
          std::vector<std::vector<float>>(
              output_logits[0].size(),
              std::vector<float>(output_logits[0][0].size(), 0.0)));

      // Compute gradients
      for (int i = 0; i < batch_size; i++) {
        assert(output_logits[i].size() == target_data_batched[i].size());
        for (int j = 0; j < output_logits[0].size(); j++) {
          grad_output[i][j][target_data_batched[i][j]] =
              -1.0 / output_logits[i][j][target_data_batched[i][j]];
        }
      }
      std::vector<std::vector<std::vector<float>>> grad_input =
          model.backward(grad_output, output_logits);

      // Update weights
      model.update_weights(learning_rate);
      std::cout << "    Step " << step << " / " << input_data.size()
                << std::endl;

    } // all steps taken, end of epoch
    auto end = std::chrono::steady_clock::now();
    std::cout
        << " Elapsed = "
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << "[sec]" << std::endl;
  }

  // TODO Save the model
  return 0;
}
