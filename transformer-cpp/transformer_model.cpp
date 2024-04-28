#include "transformer_model.hpp"
#include "matrix_utils_alias.hpp"
#include "positional_encoding.hpp"
#include <iostream>
#include <sys/wait.h>

Transformer::Transformer(const YAML::Node &config) {
  /*
   * Constructor for the Transformer model
   *
   * @param config: A YAML node containing the configuration parameters for the
   * model and the cuda kernels
   */

  num_layers = config["num_layers"].as<int>();
  batch_size = config["batch_size"].as<int>();
  seq_len = config["seq_len"].as<int>();
  vocab_size = config["vocab_size"].as<int>();
  embedding_dim = config["embedding_dim"].as<int>();
  std::vector<int> ff_hidden_dims =
      config["ff_hidden_dims"].as<std::vector<int>>();
  std::vector<int> head_sizes = config["head_sizes"].as<std::vector<int>>();
  std::vector<int> layer_output_dims =
      config["layer_output_dims"].as<std::vector<int>>();

  // Initialize the transformer layers
  // First layer takes in the input embeddings and the rest take in the output
  // of the previous layer First layer also takes in the positional encodings

  layers.push_back(TransformerLayer(seq_len, embedding_dim, head_sizes[0],
                                    ff_hidden_dims[0], layer_output_dims[0]));

  // Middle layers until the second to last layer
  for (int i = 1; i < num_layers - 1; i++) {
    layers.push_back(TransformerLayer(seq_len, layer_output_dims[i - 1],
                                      head_sizes[i], ff_hidden_dims[i],
                                      layer_output_dims[i]));
  }

  // Last layer
  // The last layer outputs vector of size vocab_size
  // This is then passed through a softmax layer to get the final output
  layers.push_back(TransformerLayer(
      seq_len, layer_output_dims[num_layers - 2], head_sizes[num_layers - 1],
      ff_hidden_dims[num_layers - 1], vocab_size));

  // Initialize the positional encodings
  positional_embeddings = PositionalEncoding::generate(seq_len, embedding_dim);
}

void Transformer::apply_positional_encoding(
    std::vector<std::vector<std::vector<float>>> &input) {
  // Apply positional encoding to the input embeddings

  for (int b = 0; b < batch_size; ++b) {
    input[b] = MatrixUtils::matrixAddition(input[b], positional_embeddings);
  }
}

std::vector<std::vector<std::vector<float>>> Transformer::forward(
    const std::vector<std::vector<std::vector<float>>> &input) {
  /*
   * Forward pass of the transformer model
   *
   * @param input: A batch of sequences of word embeddings, (batch_size,
   * seq_len, embedding_dim)
   * @return: A batch of sequences of word probabilities, (batch_size, seq_len,
   * vocab_size)
   */
  std::vector<std::vector<std::vector<float>>> input_encodings = input;
  // Add positional encodings to the input
  apply_positional_encoding(input_encodings);

  // Forward pass through each layer
  std::vector<std::vector<std::vector<float>>> output = input_encodings;
  for (int i = 0; i < num_layers; i++) {
    output = layers[i].forward(output);
  }

  // Apply softmax to the output of the last layer
  std::vector<std::vector<std::vector<float>>> output_logits =
      MatrixUtils::rowSoftmax(output);

  return output_logits;
}

std::vector<std::vector<std::vector<float>>> Transformer::backward(
    const std::vector<std::vector<std::vector<float>>> &grad_output,
    const std::vector<std::vector<std::vector<float>>> &output_logits) {
  /*
   * Backward pass of the transformer model
   *
   * @param grad_output: The gradients of the loss with respect to the output of
   * the model, (batch_size, seq_len, vocab_size)
   * @return: The gradients of the loss with respect to the input of the model,
   * (batch_size, seq_len, embedding_dim)
   */
  std::vector<std::vector<std::vector<float>>> grad =
      MatrixUtils::rowSoftmaxDerivative(grad_output, output_logits);
  for (int i = num_layers - 1; i >= 0; i--) {
    grad = layers[i].backward(grad);
  }
  return grad;
}

void Transformer::update_weights(float learning_rate) {
  // Update the weights of each layer
  for (int i = 0; i < num_layers; i++) {
    layers[i].update_weights(learning_rate);
  }
}
