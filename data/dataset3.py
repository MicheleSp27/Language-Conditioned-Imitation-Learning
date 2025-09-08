#This dataset class allows to have inside a batch the same number of placing and picking samples.

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

class CustomDataset3(Dataset):

    def __init__(self, data_path, time_sequence_length = 6):
        self._data_path = data_path  # Path to the dataset
        self._indexs = {}  # Dictionary with key the observation id and value the tuple (trajectory index, observation index)
        self._number_of_obs_for_trajectory = {} # Dictionary with key the trajectory index and value the number of observations in the trajectory
        self._range_observations = {} # Dictionary with key a traj id and value a list of observations ids
        self._time_sequence_length = time_sequence_length # Length of the history of observations

        observation_id = 0  # Observation counter
        trajectory_id = 0  # Trajectory counter
        number_of_trajectories = len(os.listdir(self._data_path)) # Number of trajectories
        set_of_trajectories = 0 # Increased after 50 trajectories

        # lenghts = 0
    

        for i in range(number_of_trajectories):

            placing = False

            """
            if i == 3:
                print("STOP!")
                break
            """
            
            
            # traj_path = data_path + "traj{}".format(i)
            # data = torch.load(traj_path)
            with (open(self._data_path + "traj{}.pkl".format(i), "rb")) as f:
                data = pickle.load(f)
            episode = data["steps"]

            if i % 100 == 0:
                if i == 0:
                    set_of_trajectories = 1
                else:
                    set_of_trajectories = set_of_trajectories + 2


            if self._number_of_obs_for_trajectory.get(set_of_trajectories, 0) == 0:
                self._number_of_obs_for_trajectory[set_of_trajectories] = 0
                self._range_observations[set_of_trajectories] = []

            if self._number_of_obs_for_trajectory.get(set_of_trajectories + 1, 0) == 0:
                self._number_of_obs_for_trajectory[set_of_trajectories + 1] = 0
                self._range_observations[set_of_trajectories + 1] = []
 
            # lenghts += len(episode)

            for observation_index in range(len(episode)):

                if episode[observation_index]["action"]["gripper_closedness_action"][0] == 1:
                    placing = True
                
                if placing == False :
                    #Picking Phase
                    self._indexs[observation_id] = (trajectory_id, observation_index)
                    self._range_observations[set_of_trajectories].append(observation_id)
                    observation_id = observation_id + 1
                else :
                    #Placing Phase
                    self._indexs[observation_id] = (trajectory_id, observation_index)
                    self._range_observations[set_of_trajectories + 1].append(observation_id)
                    observation_id = observation_id + 1

            trajectory_id = trajectory_id + 1

            # Updating picking samples
            total_picking_samples = len(self._range_observations[set_of_trajectories])
            added_picking_samples = self._number_of_obs_for_trajectory[set_of_trajectories]
            non_added_picking_samples = total_picking_samples - added_picking_samples
            self._number_of_obs_for_trajectory[set_of_trajectories] += non_added_picking_samples
            # Updating placing samples
            total_placing_samples = len(self._range_observations[set_of_trajectories + 1])
            added_placing_samples = self._number_of_obs_for_trajectory[set_of_trajectories + 1]
            non_added_placing_samples = total_placing_samples - added_placing_samples
            self._number_of_obs_for_trajectory[set_of_trajectories + 1] += non_added_placing_samples
            """
            print("Number of samples in the trajectory {}".format(len(episode)))
            print("Picking samples in the trajectory {} : {}".format(i,non_added_picking_samples))
            print("Placing samples in the trajectory {} : {}".format(i, non_added_placing_samples))
            print("List of picking samples : {}".format(self._range_observations[set_of_trajectories]))
            print("List of placing samples : {}".format(self._range_observations[set_of_trajectories + 1]))
            print("Total Number of samples seen {}".format(lenghts))
            print("Total Number of picking samples {}".format(self._number_of_obs_for_trajectory[set_of_trajectories]))
            print("Total Number of placing samples {}".format(self._number_of_obs_for_trajectory[set_of_trajectories + 1]))
            
        for i in range(1, 32+1):

            low_bound = self._range_observations[i][0]
            high_bound = self._range_observations[i][-1]

            print("For trajectory {} the lower bound is {} while the higher bound is {}".format(i, low_bound, high_bound))
        """
        """
        for i in range(1,32+1):
            number_of_observations = self._number_of_obs_for_trajectory[i]
            self._range_observations[i] = []

            if i == 1:
                low_bound = 0
                high_bound = number_of_observations
            else :
                low_bound = self._range_observations[i-1][-1] + 1
                high_bound = low_bound + number_of_observations
            
            print("For trajectory {} the lower bound is {} while the higher bound is {}".format(i, low_bound, high_bound))

            for j in range(low_bound, high_bound):
                self._range_observations[i].append(j)
        """
                

        print("Number of trajectories : {}".format(trajectory_id))
        print("Number of observations : {}".format(observation_id))
        print("Number of observations for each trajectory : {}".format(self._number_of_obs_for_trajectory))
        
        
        """
        print(self._number_of_obs_for_trajectory)
        
        
        for i in range(1,32+1):
            lower_bound = self._range_observations[i][0]
            higher_bound = self._range_observations[i][-1]
            print("For trajectory {} the lower bound is {} while the higher bound is {}".format(i, lower_bound, higher_bound))
        """


    def __len__(self):
       pass

    def __getitem__(self,idx):

        trajectory_index, observation_index = self._indexs[idx] # Retrieving the trajectory and observation index from the observation id
        with(open(self._data_path + "traj{}.pkl".format(trajectory_index), "rb")) as f:
            data = pickle.load(f)
        # data = torch.load(self._data_path + "traj{}".format(trajectory_index))
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
                """
                action_gripper = episode[k]["action"]["gripper_closedness_action"][None,:]
                action_rotation_delta = episode[k]["action"]["rotation_delta"][None,:]
                action_world_vector = episode[k]["action"]["world_vector"][None,:]
                """
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
                """
                next_action_gripper = episode[k]["action"]["gripper_closedness_action"][None,:]
                next_action_rotation_delta = episode[k]["action"]["rotation_delta"][None,:]
                next_action_world_vector = episode[k]["action"]["world_vector"][None,:]
                """
                next_action_gripper = episode[k]["action"]["gripper_closedness_action"][None,:]                                
                next_action_x_axis = np.array([episode[k]["action"]["world_vector"][0]])[None,:]
                next_action_y_axis = np.array([episode[k]["action"]["world_vector"][1]])[None,:]
                next_action_z_axis = np.array([episode[k]["action"]["world_vector"][2]])[None,:]
                next_action_roll = np.array([episode[k]["action"]["rotation_delta"][0]])[None,:]
                next_action_pitch = np.array([episode[k]["action"]["rotation_delta"][1]])[None,:]
                next_action_yaw = np.array([episode[k]["action"]["rotation_delta"][2]])[None,:]


                images = torch.concatenate((images, next_images), dim = 0)
                natural_language_embedding = np.concatenate((natural_language_embedding, next_natural_language_embedding), axis = 0)
                """
                action_gripper = np.concatenate((action_gripper, next_action_gripper), axis = 0)
                action_rotation_delta = np.concatenate((action_rotation_delta, next_action_rotation_delta), axis = 0)
                action_world_vector = np.concatenate((action_world_vector, next_action_world_vector), axis = 0)
                """
                action_gripper = np.concatenate((action_gripper, next_action_gripper), axis = 0)
                action_x_axis = np.concatenate((action_x_axis, next_action_x_axis), axis = 0)
                action_y_axis = np.concatenate((action_y_axis, next_action_y_axis), axis = 0)
                action_z_axis = np.concatenate((action_z_axis, next_action_z_axis), axis = 0)
                action_roll = np.concatenate((action_roll, next_action_roll), axis = 0)
                action_pitch = np.concatenate((action_pitch, next_action_pitch), axis = 0)
                action_yaw = np.concatenate((action_yaw, next_action_yaw), axis = 0)


        return images, natural_language_embedding, action_gripper, action_x_axis, action_y_axis, action_z_axis, action_roll, action_pitch, action_yaw
    


if __name__ == "__main__":
    dataset = CustomDataset3("/mnt/localstorage/mspremulli/Datasets/Simulated_Converted_Delta2/")
    
