#include "transformer_layer.hpp"
#include "matrix_utils.hpp"
#include "positional_encoding.hpp"

TransformerLayer::TransformerLayer(int input_dim, int embedding_dim, int head_size, int ff_hidden_dim, int output_dim) : 
    self_attention(embedding_dim, head_size), 
    feed_forward(head_size, ff_hidden_dim, output_dim), 
    input_dim_(input_dim), embedding_dim_(embedding_dim),
    head_size_(head_size), ff_hidden_dim_(ff_hidden_dim), output_dim_(output_dim) 
{}


std::vector<std::vector<std::vector<float>>> TransformerLayer::forward(const std::vector<std::vector<std::vector<float>>>& input) {
    /*
    * Forward pass of the transformer layer, we skip normalization and residual connections for simplicity
    *
    * @param input: input embeddings of shape (batch_size, seq_len, embedding_dim) 
    * @return: output embeddings of shape (batch_size, seq_len, output_dim)
    */ 

    self_attention_input_ = input; // input to self-attention layer, saving for backpropagation
    
    // Apply self-attention
    // Create mock mask, TODO: remove and implement properly
    std::vector<std::vector<std::vector<float>>> mask(input.size(), std::vector<std::vector<float>>(input[0].size(), std::vector<float>(input[0][0].size(), 1.0))); 
    std::vector<std::vector<std::vector<float>>> self_attention_output = self_attention.forward(input, mask);

    feed_forward_input_ = self_attention_output; // input to feed forward network, saving for backpropagation

    // Apply feed forward network
    std::vector<std::vector<std::vector<float>>> output = feed_forward.forward(self_attention_output);
    return output;
}

std::vector<std::vector<std::vector<float>>> TransformerLayer::backward(const std::vector<std::vector<std::vector<float>>>& grad_output) {
    /*
    * Backward pass of the transformer layer
    * 
    * @param grad_output: gradient of the loss w.r.t. the output of the transformer layer
    * @return: gradient of the loss w.r.t. the input of the transformer layer
    */ 

    // Backward pass of the feed forward network
    std::vector<std::vector<std::vector<float>>> grad_feed_forward = feed_forward.backward(feed_forward_input_, grad_output); // gradient of the loss w.r.t. the input of the feed forward network, shape (batch_size, seq_len, output_dim)

    // Backward pass of the self-attention layer
    std::vector<std::vector<std::vector<float>>> grad_self_attention = self_attention.backward(self_attention_input_, grad_feed_forward); // gradient of the loss w.r.t. the input of the self-attention layer, shape (batch_size, seq_len, embedding_dim)

    return grad_self_attention; // gradient of the loss w.r.t. the input of the transformer layer, shape (batch_size, seq_len, embedding_dim)
}


void TransformerLayer::update_weights(float learning_rate) {
    // Update the weights of the self-attention and feed forward layers
    self_attention.update_weights(learning_rate);
    feed_forward.update_weights(learning_rate);
}

