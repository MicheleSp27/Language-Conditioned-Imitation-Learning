# Dataset custom class for the RT-1 Original Real Dataset Pre-Training. 
import tensorflow as tf
import tensorflow_datasets as tfds
from torch.utils.data import Dataset
import torch 
import numpy as np
import hydra
import pickle
import os
from torchvision.transforms import ToTensor


class CustomDataset(Dataset):

  def __init__(self, data_path, time_sequence_length, discard_length):

    # Setting parameters.
    self._data_path = data_path  # Path to the dataset
    self._time_sequence_length = time_sequence_length  # Length of the history of observations 
    self._discard_length = discard_length # Value that indicates the length of the trajectories that will be discarded. In the dataset there are trajectories with a small temporal length. 
            
    # Data Definition
    self._trajectory_index = 0 # Trajectory counter
    self._observation_id = 0 # Observation counter
    self._indexs = {} # Dictionary with key the observation id and value the tuple (trajectory index, observation index)
                      # where trajectory index is the trajectory to which the observation belongs and observation index is the index of the observation in the trajectory
    self._trajectory_obs_id = {} # Dictionary with key the trajectory index and value a list with the observation ids of the trajectory
    self._number_of_obs_for_trajectory = {} # Dictionary with key the trajectory index and value the number of observations in the trajectory

    # Data Analysis 
    
    self._number_of_discarded_trajectories = 0 # Number of discarded trajectories with length < discard_length
    self._number_of_discarded_obs = 0 # Number of discarded observations from trajectories with length < discard_length
    
    # Number of trajectories in the dataset
    number_of_trajectories = len(os.listdir(self._data_path))
  
    # Data processing
    for index in range(number_of_trajectories):

      # Load the trajectory data with pickle
      with open(self._data_path + "traj{}".format(index), "rb") as f:
        data = pickle.load(f)

      episode = data["steps"] # Trajectory Data is saved in a dictionary with "steps" key

      # Discarding trajectories with length < discard_length
      if len(episode) < self._discard_length:
        self._trajectory_index += 1
        self._number_of_discarded_trajectories += 1
        self._number_of_discarded_obs += len(episode)
        continue
      
      # Adding the trajectory to the avalaible ones
      self._trajectory_obs_id[self._trajectory_index] = []
      self._number_of_obs_for_trajectory[self._trajectory_index] = len(episode)
      
      # Updating data structures
      for observation_index in range(len(episode)):

        self._indexs[self._observation_id] = (self._trajectory_index, observation_index)
        self._trajectory_obs_id[self._trajectory_index].append(self._observation_id)
        self._observation_id = self._observation_id + 1

      self._trajectory_index = self._trajectory_index + 1
    
    print("The number of trajectories in the dataset is : {}".format(number_of_trajectories))
    print("The number of training trajectories is : {}".format(len(self._number_of_obs_for_trajectory)))
    print("The number of training observations is : {}".format(self._observation_id))
    print("The number of discarded trajectories due to temporal length smaller than {} is : {}".format(self._time_sequence_length, self._number_of_discarded_trajectories))
    print("The number of discarded observations due to temporal length smaller than {} is : {}".format(self._time_sequence_length, self._number_of_discarded_obs))
      
  def __len__(self):

    pass

  def __getitem__(self,idx):
        
      trajectory_index, observation_index = self._indexs[idx] # Retrieving the trajectory and observation index from the observation id
      
      # Retrieving the trajectory 
      with open(self._data_path + "traj{}".format(trajectory_index), "rb") as f:
        data = pickle.load(f)
      episode = data["steps"]

      # Extracting the history of observations from the trajectory
      # The lenght of the history is equal to time_sequence_length
      # Observations from index observation_index - time_sequence_length + 1 to observation_index are extracted.
      # If observation_index - time_sequence_length + 1 < 0, the missing observations are replaced with the first observation of the trajectory at index 0.
      # For example, if observation_index = 2 and time_sequence_lenght = 6, the following state index are extracted:
      # [0,0,0,0,1,2], the observation at index 0 is repeated four times in this case. 

    
      low_index = observation_index - self._time_sequence_length + 1

      for i in range(low_index, observation_index + 1):

        if i < 0 :
          k = 0
        else :
          k = i

        # Data Concatenation at time step dimension

        if i == low_index:

          images = episode[k]["observation"]["image"][None,:,:,:]
          natural_language_embedding = episode[k]["observation"]["natural_language_embedding"][None,:]
          action_gripper = episode[k]["action"]["gripper_closedness_action"][None,:]
          action_rotation_delta = episode[k]["action"]["rotation_delta"][None,:]
          action_world_vector = episode[k]["action"]["world_vector"][None,:]

        else:

          next_images = episode[k]["observation"]["image"][None,:,:,:]
          next_natural_language_embedding = episode[k]["observation"]["natural_language_embedding"][None,:]
          next_action_gripper = episode[k]["action"]["gripper_closedness_action"][None,:]
          next_action_rotation_delta = episode[k]["action"]["rotation_delta"][None,:]
          next_action_world_vector = episode[k]["action"]["world_vector"][None,:]
         

          images = torch.concatenate((images, next_images), dim = 0)
          natural_language_embedding = np.concatenate((natural_language_embedding, next_natural_language_embedding), axis = 0)
          action_gripper = np.concatenate((action_gripper, next_action_gripper), axis = 0)
          action_rotation_delta = np.concatenate((action_rotation_delta, next_action_rotation_delta), axis = 0)
          action_world_vector = np.concatenate((action_world_vector, next_action_world_vector), axis = 0)

      return images, natural_language_embedding, action_gripper, action_rotation_delta, action_world_vector
  
    
     
if __name__ == "__main__" :   

  dataset = CustomDataset(data_path = "", time_sequence_length = 6, discard_length = 12)
  
