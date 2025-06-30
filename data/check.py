import tensorflow_hub as hub
import numpy as np
import torch

if __name__ == '__main__' :

    embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/large/2")
    language_instruction_dict = {"task_0" : "Pick green box and place it into the first bin", "task_1" : "Pick green box and place it into the second bin", "task_2" : "Pick green box and place it into the third bin", "task_3" : "Pick green box and place it into the fourth bin", "task_4" : "Pick yellow box and place it into the first bin", "task_5" : "Pick yellow box and place it into the second bin", "task_6" : "Pick yellow box and place it into the third bin", "task_7" : "Pick yellow box and place it into the fourth bin", "task_8" : "Pick blue box and place it into the first bin", "task_9" : "Pick blue box and place it into the second bin", "task_10" : "Pick blue box and place it into the third bin", "task_11" : "Pick blue box and place it into the fourth bin", "task_12" : "Pick red box and place it into the first bin", "task_13" : "Pick red box and place it into the second bin", "task_14" : "Pick red box and place it into the third bin", "task_15" : "Pick red box and place it into the fourth bin"} # Language Instruction Dictionary
    language_embedding = '/user/mspremulli/Language-Conditioned-Imitation-Learning/data/language_embeddings.pkl'

    language_embeddings = torch.load(language_embedding) # Load the embeddings
    
    for task in language_instruction_dict:

        print("Current Task: {}".format(task))

        task_id = int(task.split('_')[1])

        # Embedding of the dataset
        instruction = language_instruction_dict[task] # Get the instruction
        embedding = embed([instruction]) # The model wants as input a list of strings
        embedding = embedding.numpy() # Convert to numpy array
        embedding = np.squeeze(embedding, axis = 0) # Removing the first dimensione
        print(embedding.shape)

        # Embedding for inference
        inference_embedding = language_embeddings[task_id] # Get the embedding for the task
        print(inference_embedding.shape)

        # Check if the embeddings are equal
        
        print("The embeddings are {}".format(np.array_equal(embedding, inference_embedding)))

