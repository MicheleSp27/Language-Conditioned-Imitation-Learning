import tensorflow_hub as hub # Used to Load the universal sentence encoder for natural language processing
import torch
import numpy as np

if __name__ == "__main__":

    # Load the universal sentence encoder
    embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/large/2") # Loading the universal sentence encoder 
    instruction1 = "Pick greenbox and place it into the first bin" # Language Instruction 
    instruction2 = "Pick redbox and place it into the first bin" # Language Instruction 
    
    natural_language_embedding1 = embed([instruction1])
    natural_language_embedding1 = natural_language_embedding1.numpy() # Conversion from tensorflow.python.framework.ops.EagerTensor' to numpy as the original dataset
    natural_language_embedding1 = np.squeeze(natural_language_embedding1, axis = 0) # From (1, 512, ) to (512, ) like the original dataset
    natural_language_embedding1 = torch.tensor(natural_language_embedding1) # Because they are used during the inference, they must be tensors

    natural_language_embedding2 = embed([instruction2])
    natural_language_embedding2 = natural_language_embedding2.numpy() # Conversion from tensorflow.python.framework.ops.EagerTensor' to numpy as the original dataset
    natural_language_embedding2 = np.squeeze(natural_language_embedding2, axis = 0) # From (1, 512, ) to (512, ) like the original dataset
    natural_language_embedding2 = torch.tensor(natural_language_embedding2) # Because they are used during the inference, they must be tensors
    # Calculate the cosine similarity
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    similarity = cos(natural_language_embedding1, natural_language_embedding2)
    print("Cosine Similarity: ", similarity.item())