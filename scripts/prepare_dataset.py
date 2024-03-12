from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import torch
import sys
import os
import tqdm


EMBEDDINGS_FILE = "../dataset/embeddings.csv"
INPUT_IDS_FILE = "../dataset/input_ids.csv"

def chunk_text(text, tokenizer, max_length):
    # Tokenize the text to ids
    tokens = tokenizer.encode(text)
    
    # Calculate the number of chunks needed
    num_chunks = len(tokens) // (max_length - 1) + 1
    
    # Split tokens into chunks
    chunks = [
        tokens[i * (max_length - 1):(i + 1) * (max_length - 1)] for i in range(num_chunks)
    ]
    
    return chunks

if __name__ == "__main__":
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    # Ensure the output directories exist
    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    
    # Read argument from command line
    text_file = sys.argv[1]
    if len(sys.argv) > 2:
        max_length = int(sys.argv[2])
    else:
        max_length = 1024

    # Read text from the file
    with open(text_file, "r") as file:
        text = file.read()

    # Chunk the text
    chunks = chunk_text(text, tokenizer, max_length)
    
    # Initialize lists to store combined input ids and embeddings
    combined_input_ids = []
    combined_embeddings = []
    
    for chunk in tqdm.tqdm(chunks):
        # Convert chunk to tensor
        input_ids_tensor = torch.tensor([chunk])

        # Generate embeddings
        with torch.no_grad():
            outputs = model(input_ids_tensor)
            embeddings = outputs.last_hidden_state.squeeze().detach().numpy()
            
            combined_input_ids.extend(chunk)
            combined_embeddings.append(embeddings)
    
    # Concatenate embeddings from all chunks
    embeddings_concat = np.concatenate(combined_embeddings, axis=0)
    
    # Save embeddings and input_ids
    np.savetxt(EMBEDDINGS_FILE, embeddings_concat, delimiter=",")
    np.savetxt(INPUT_IDS_FILE, [combined_input_ids], delimiter=",", fmt="%d")

    print(f"Vocabulary size: {len(tokenizer)}\n")
    print(f"Processed text split into {len(chunks)} chunks.")
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")
    print(f"Input IDs saved to {INPUT_IDS_FILE}")
    print(f"Embeddings shape: {embeddings_concat.shape}")
    print(f"Input IDs length: {len(combined_input_ids)}")
    assert len(combined_input_ids) == embeddings_concat.shape[0]
