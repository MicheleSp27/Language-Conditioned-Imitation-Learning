import tensorflow as tf
import tensorflow_datasets as tfds
from torch.utils.data import Dataset
import torch 
import numpy as np
import hydra
import pickle
import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional
from torchvision.transforms import ToTensor
from PIL import Image

tf.config.set_visible_devices([], 'GPU')

class CustomDataset(Dataset):

  def __init__(self, data_path, time_sequence_length, discard_length):

    self._data_path = data_path  # Path to the dataset
    self._time_sequence_length = time_sequence_length  # Length of the history of observations 
    self._discard_length = discard_length # Value that indicates the length of the trajectories that will be discarder
    self._to_tensor = ToTensor() # ToTensor transformation
            
    # Data Definition

    self._trajectory_index = 0 # Trajectory counter
    self._observation_id = 0 # Observation counter
    self._indexs = {} # Dictionary with key the observation id and value the tuple (trajectory index, observation index)
                      # where trajectory index is the trajectory to which the observation belongs and observation index is the index of the observation in the trajectory
    self._trajectory_obs_id = {} # Dictionary with key the trajectory index and value a list with the observation ids of the trajectory
    self._number_of_obs_for_trajectory = {} # Dictionary with key the trajectory index and value the number of observations in the trajectory
    self._available_trajectories = [] # List with indexs of the available trajectories

    # Data Analysis 
    
    self._number_of_discarded_obs = 0 # Number of discarded observations
    self._number_of_discarded_trajectories = 0 # Number of discarded trajectories

    number_of_trajectories = len(os.listdir(self._data_path))
    print("Number of trajectories : {}.".format(number_of_trajectories))
  
    # Data Initialization
    for index in range(number_of_trajectories):

      # Stop



      data = torch.load(data_path + "traj{}".format(index))
      episode = data["steps"] # Trajectory Data

      # Discarding trajectories with length less than discard_length
      if len(episode) < self._discard_length:
        self._trajectory_index = self._trajectory_index + 1
        self._number_of_discarded_trajectories = self._number_of_discarded_trajectories + 1
        self._number_of_discarded_obs = self._number_of_discarded_obs + len(episode)
        continue
      
      self._available_trajectories.append(self._trajectory_index)
      self._trajectory_obs_id[self._trajectory_index] = []
      self._number_of_obs_for_trajectory[self._trajectory_index] = len(episode)
      
      
      for observation_index in range(len(episode)):

        self._indexs[self._observation_id] = (self._trajectory_index, observation_index)
        self._trajectory_obs_id[self._trajectory_index].append(self._observation_id)
        self._observation_id = self._observation_id + 1

      self._trajectory_index = self._trajectory_index + 1
    
    print("The number of trajectories in the dataset is : {}".format(self._trajectory_index))
    print("The number of discarded trajectories due to small temporal length is : {}".format(self._number_of_discarded_trajectories))
    print("The number of discarded observations is : {}".format(self._number_of_discarded_obs))
      
  def __len__(self):

    return self._observation_id

  def __getitem__(self,idx):
        
      trajectory_index, observation_index = self._indexs[idx] # Retrieving the trajectory and observation index from the observation id
      data = torch.load(self._data_path + "traj{}".format(trajectory_index))
      episode = data["steps"]

      # Extracting the history of observations from the trajectory
      # The lenght of the history is equal to time_sequence_length
      # Observations from observation_index - time_sequence_length + 1 to observation_index are extracted
      # If observation_index - time_sequence_length + 1 < 0, the missing observations are replaced with the first observation of the trajectory


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
  

@hydra.main(version_base=None, config_path="", config_name="config")
def dataset_analysis(cfg):
  
  print(cfg)
  custom_Dataset = hydra.utils.instantiate(cfg.rt1_dataset)
  
  print("Number of Trajectories : {}".format(custom_Dataset._trajectory_index))
  print("Number of Trajectories Discarded : {}".format(custom_Dataset._number_of_discarded_trajectories))
  print("Number of Observations : {}".format(len(custom_Dataset)))
  print("Number of Observations Discarded : {}".format(custom_Dataset._number_of_discarded_obs))
  print("Longhest Trajectory : {}".format(custom_Dataset._longhest_trajectory))
  print("Index of the Longhest Trajectory : {}".format(custom_Dataset._index_longhest_trajectory))

  file = open("trajectory_length.txt", "w")
  file.write("The total number of trajectories is: " + str(custom_Dataset._trajectory_index) + "\n")
  file.write("The total number of observations is: " + str(len(custom_Dataset)) + "\n")
  file.write("The longhest trajectory is: " + str(custom_Dataset._longhest_trajectory) + "\n")
  file.write("The index of the longhest trajectory is: " + str(custom_Dataset._index_longhest_trajectory) + "\n")

  for i in range(1, custom_Dataset._longhest_trajectory + 1):
    if custom_Dataset._analysis.get(i,0) != 0:
      file.write("Number of trajectories with length " + str(i) + " : " + str(custom_Dataset._analysis[i]) + "\n")
  file.close()
    
     
if __name__ == "__main__" :   
  # Dataset Analysis  
  dataset_analysis()
  # dataset = CustomDataset(data_path = "/mnt/localstorage/mspremulli/RT-1_Dataset/rt-1-data-release/", time_sequence_length = 6, image_channels = 3, image_height = 256, image_width = 320, token_size = 512, original_dataset = True, discard_length = 12)
  
