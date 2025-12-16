# Script used to save natural language embeddings into pkl files.

import tensorflow_hub as hub # Used to Load the universal sentence encoder for natural language processing. This is the language encoder used in the original RT-1 work.
import torch
import numpy as np

if __name__ == "__main__":

    embeddings_name ="language_embeddings"
    # Load the universal sentence encoder
    embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/large/2") # Loading the universal sentence encoder.
    language_instruction_dict = {0 : "Pick green box and place it into the first bin", 1 : "Pick green box and place it into the second bin", 2 : "Pick green box and place it into the third bin", 3 : "Pick green box and place it into the fourth bin", 4 : "Pick orange box and place it into the first bin", 5 : "Pick orange box and place it into the second bin", 6 : "Pick orange box and place it into the third bin", 7 : "Pick orange box and place it into the fourth bin", 8 : "Pick red box and place it into the first bin", 9 : "Pick red box and place it into the second bin", 10 : "Pick red box and place it into the third bin", 11 : "Pick red box and place it into the fourth bin", 12 : "Pick grey box and place it into the first bin", 13 : "Pick grey box and place it into the second bin", 14 : "Pick grey box and place it into the third bin", 15 : "Pick grey box and place it into the fourth bin"} # Language Instruction Dictionary.
    embeddings_dict = {} # Dictionary that store var_id-natural language embedding pairs.

    for var_id in language_instruction_dict.keys():

        language_instruction = language_instruction_dict[var_id] # Retrieving the language instruction.
        natural_language_embedding = embed([language_instruction]) # Compute the embedding.
        natural_language_embedding = natural_language_embedding.numpy() # Conversion from tensorflow.python.framework.ops.EagerTensor' to numpy as the original dataset
        natural_language_embedding = np.squeeze(natural_language_embedding, axis = 0) # From (1, 512, ) to (512, ) like the original dataset
        natural_language_embedding = torch.tensor(natural_language_embedding) # Because they are used during the inference, they must be tensors
        embeddings_dict[var_id] = natural_language_embedding # Storing the embedding
    
    torch.save(embeddings_dict, embeddings_name + ".pkl") # Save the embeddings in to pkl file