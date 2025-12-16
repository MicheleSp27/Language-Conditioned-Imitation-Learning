# This dataset class allows to have inside a batch the same number of placing and picking samples.
# Also it is possible to pass in input tasks as list and trajectories set as json to be excluded from the training data.
# Dataset Custom class for the simulated MIVIA Dataset
from torch.utils.data import Dataset
import torch 
import numpy as np
import hydra
import pickle
import os
import json
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional
from torchvision.transforms import ToTensor
from PIL import Image

class CustomDataset2(Dataset):

    def __init__(self, data_path, excluded_tasks, trajectories_set, time_sequence_length = 6):
        
        self._data_path = data_path  # Path to the dataset
        self._indexs = {}  # Dictionary with key the observation id and value the tuple (trajectory index, observation index)
        self._number_of_obs_for_trajectory = {} # Dictionary with key the trajectory index and value the number of observations in the trajectory
        self._range_observations = {} # Dictionary with key a traj id and value a list of observations ids
        self._time_sequence_length = time_sequence_length # Length of the history of observations
        self._excluded_tasks = excluded_tasks # List containing the tasks that are excluded from the training procedure. This allows for generalization testing.
        self._traj_index_folder_map = {} # Dictionary storing mapping between trajectory and folder in which are stored.
        self._trajectories_set = trajectories_set # Path to the json set of trajectories to be used for the training.

        observation_id = 0  # Observation counter
        trajectory_id = 0  # Trajectory counter
        available_trajectories = 0
        number_of_tasks = len(os.listdir(self._data_path)) # Number of available Tasks
        set_of_trajectories = 0 # Used to aggregate trajectories

        # Loading the set of available trajectories
        if self._trajectories_set:
            with open(self._trajectories_set, "r") as f:
                traj_set = json.load(f)
        else:
            traj_set = None
    
        # Reading trajectories from the data folder in order
        for i in range(0, number_of_tasks):

            task = "task_{}".format(i)

            # Check if the task must be part of the training data.
            if task in self._excluded_tasks:
                trajectory_id += 100 # Skip task
            else:
                if set_of_trajectories == 0:
                    set_of_trajectories = 1
                else :
                    set_of_trajectories += 2

                for traj_index in range(0, 100):

                    if traj_index < 10:
                        traj_path = "traj00{}.pkl".format(traj_index)
                    else:
                        traj_path = "traj0{}.pkl".format(traj_index)

                    # Check if the current trajectory is part of the allowed ones
                    if traj_set is not None and traj_path not in traj_set[task]:
                        trajectory_id += 1 # Skip trajectory
                        continue
                    else:

                        placing = False
                        self._traj_index_folder_map[trajectory_id] = i
                        with (open(self._data_path + "{}/{}".format(task, traj_path), "rb")) as f:
                            data = pickle.load(f)
                        episode = data["steps"]

                        # Separing picking and placing samples
                        if self._number_of_obs_for_trajectory.get(set_of_trajectories, 0) == 0:
                            self._number_of_obs_for_trajectory[set_of_trajectories] = 0
                            self._range_observations[set_of_trajectories] = []

                        if self._number_of_obs_for_trajectory.get(set_of_trajectories + 1, 0) == 0:
                            self._number_of_obs_for_trajectory[set_of_trajectories + 1] = 0
                            self._range_observations[set_of_trajectories + 1] = []

                        for observation_index in range(len(episode)):
                            
                            # Placing phase is active when the gripper is closed(action value of 1)
                            if episode[observation_index]["action"]["gripper_closedness_action"][0] == 1:
                                placing = True
                            
                            if placing == False :
                                # Picking sample
                                self._indexs[observation_id] = (trajectory_id, observation_index)
                                self._range_observations[set_of_trajectories].append(observation_id)
                                observation_id = observation_id + 1
                            else :
                                # Placing sample
                                self._indexs[observation_id] = (trajectory_id, observation_index)
                                self._range_observations[set_of_trajectories + 1].append(observation_id)
                                observation_id = observation_id + 1

                        trajectory_id = trajectory_id + 1
                        available_trajectories +=1

                        # Updating number of picking samples
                        total_picking_samples = len(self._range_observations[set_of_trajectories])
                        added_picking_samples = self._number_of_obs_for_trajectory[set_of_trajectories]
                        non_added_picking_samples = total_picking_samples - added_picking_samples
                        self._number_of_obs_for_trajectory[set_of_trajectories] += non_added_picking_samples
                        # Updating number of placing samples
                        total_placing_samples = len(self._range_observations[set_of_trajectories + 1])
                        added_placing_samples = self._number_of_obs_for_trajectory[set_of_trajectories + 1]
                        non_added_placing_samples = total_placing_samples - added_placing_samples
                        self._number_of_obs_for_trajectory[set_of_trajectories + 1] += non_added_placing_samples
                
        print("The number of trajectories in the dataset is : {}".format(trajectory_id))
        print("The number of training trajectories is : {}".format(available_trajectories))
        print("The number of training observations is : {}".format(observation_id))
        print("The number of discarded trajectories is : {}".format(trajectory_id - available_trajectories))

    def __len__(self):
       pass

    def __getitem__(self,idx):

        trajectory_index, observation_index = self._indexs[idx] # Retrieving the trajectory and observation index from the observation id
        task_folder_index = self._traj_index_folder_map[trajectory_index] # Retrieving var_id from the trajectory index
        # Computing the trajectory index for the var_id
        if task_folder_index > 0:
            task_trajectory = trajectory_index % (100 * task_folder_index)
        else:
            task_trajectory = trajectory_index
        if task_trajectory < 10:
            traj_path = "traj00{}.pkl".format(task_trajectory)
        else:
            traj_path = "traj0{}.pkl".format(task_trajectory)
        # Reading trajectory data
        with open(self._data_path + "task_{}/{}".format(task_folder_index, traj_path), "rb") as f:
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
                action_x_axis = np.array([episode[k]["action"]["world_vector"][0]])[None,:]
                action_y_axis = np.array([episode[k]["action"]["world_vector"][1]])[None,:]
                action_z_axis = np.array([episode[k]["action"]["world_vector"][2]])[None,:]
                action_roll = np.array([episode[k]["action"]["rotation_delta"][0]])[None,:]
                action_pitch = np.array([episode[k]["action"]["rotation_delta"][1]])[None,:]
                action_yaw = np.array([episode[k]["action"]["rotation_delta"][2]])[None,:]

            else:

                next_images = episode[k]["observation"]["image"][None,:,:,:]
                next_natural_language_embedding = episode[k]["observation"]["natural_language_embedding"][None,:]
                next_action_gripper = episode[k]["action"]["gripper_closedness_action"][None,:]                                
                next_action_x_axis = np.array([episode[k]["action"]["world_vector"][0]])[None,:]
                next_action_y_axis = np.array([episode[k]["action"]["world_vector"][1]])[None,:]
                next_action_z_axis = np.array([episode[k]["action"]["world_vector"][2]])[None,:]
                next_action_roll = np.array([episode[k]["action"]["rotation_delta"][0]])[None,:]
                next_action_pitch = np.array([episode[k]["action"]["rotation_delta"][1]])[None,:]
                next_action_yaw = np.array([episode[k]["action"]["rotation_delta"][2]])[None,:]


                images = torch.concatenate((images, next_images), dim = 0)
                natural_language_embedding = np.concatenate((natural_language_embedding, next_natural_language_embedding), axis = 0)
                action_gripper = np.concatenate((action_gripper, next_action_gripper), axis = 0)
                action_x_axis = np.concatenate((action_x_axis, next_action_x_axis), axis = 0)
                action_y_axis = np.concatenate((action_y_axis, next_action_y_axis), axis = 0)
                action_z_axis = np.concatenate((action_z_axis, next_action_z_axis), axis = 0)
                action_roll = np.concatenate((action_roll, next_action_roll), axis = 0)
                action_pitch = np.concatenate((action_pitch, next_action_pitch), axis = 0)
                action_yaw = np.concatenate((action_yaw, next_action_yaw), axis = 0)


        return images, natural_language_embedding, action_gripper, action_x_axis, action_y_axis, action_z_axis, action_roll, action_pitch, action_yaw
    
if __name__ == "__main__":
    dataset = CustomDataset4("/mnt/localstorage/mspremulli/Datasets/Simulated_Converted_Delta/", ['task_0', 'task_5', 'task_10', 'task_15'])
    
