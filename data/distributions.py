# This script is used to compute the distribution of the action tokens across the datatset
import os
import numpy as np
import torch
import sys
from gym import spaces
import matplotlib.pyplot as plt
import pickle

sys.path.insert(0,'/user/mspremulli/Language-Conditioned-Imitation-Learning/')
from collections import OrderedDict
from model.action_tokenizer_cpu import RT1ActionTokenizer

def build_histogram_and_save(histogram_data, original, changed_action_space, DoF, num_bins = 256):

    # File name definition
    file_save_path = ""
    if original:
        file_save_path ="Token-Distribution-OriginalDataset-DoF={}-".format(DoF)
    else:
        if changed_action_space:
            file_save_path ="Token-Distribution-SimulatedDataset-ChangedActionSpace-DoF={}-".format(DoF)
        else:
            file_save_path ="Token-Distribution-SimulatedDataset-OriginalActionSpace-DoF={}-".format(DoF)


    file = open(file_save_path + ".txt", "w")
    if DoF == 'gripper_closedness':
        num_bins = 2
    for i in range(num_bins):
        file.write("Token {} Frequency: {}\n".format(i, histogram_data[i]))
    
    file.close()

    """
    bin_list = [i for i in range(num_bins)]
    plt.bar(bin_list, histogram_data, width=1)
    plt.title("Token Distribution for DoF: {}".format(DoF))
    plt.xlabel("Token Value")
    plt.ylabel("Frequency")
    plt.savefig(file_save_path + DoF + ".png")
    """


if __name__ == "__main__":

    data_path = "/mnt/localstorage/mspremulli/Datasets/Simulated_Converted_Delta2/"
    original = False
    changed_action_space = True

    if original:
        # EveryDay Robot action Space
        action_space = OrderedDict([
                ('world_vector', spaces.Box(low= -1.0, high= 1.0, shape=(3,), dtype=np.float32)),
                ('rotation_delta', spaces.Box(low= -3.14 / 2, high= 3.14 / 2, shape=(3,), dtype=np.float32)),
                ('gripper_closedness_action', spaces.Box(low= -1.0 , high= 1.0, shape=(1,), dtype=np.float32))
                ])
        
    else:
        # UR5e action space
        if changed_action_space:

            action_space = OrderedDict([
                    ('x_axis', spaces.Box(low= -0.007, high= 0.025, shape=(1,), dtype=np.float32)),
                    ('y_axis', spaces.Box(low= -0.016, high= 0.017, shape=(1,), dtype=np.float32)),
                    ('z_axis', spaces.Box(low= -0.018, high= 0.017, shape=(1,), dtype=np.float32)),
                    ('roll', spaces.Box(low= -0.06, high= 0.08, shape=(1,), dtype=np.float32)),
                    ('pitch', spaces.Box(low= -0.11, high= 0.05, shape=(1,), dtype=np.float32)),
                    ('yaw', spaces.Box(low= -0.08, high= 0.3, shape=(1,), dtype=np.float32)),
                    ('gripper_closedness_action', spaces.Discrete(2))])
        else:

            action_space = OrderedDict([
                ('world_vector', spaces.Box(low= -1.0, high= 1.0, shape=(3,), dtype=np.float32)),
                ('rotation_delta', spaces.Box(low= -6.28, high= 6.28, shape=(3,), dtype=np.float32)),
                ('gripper_closedness_action', spaces.Discrete(2))
                ])

    action_tokenizer = RT1ActionTokenizer(action_space, 256) # Initialize the action tokenizer
    traj_list = os.listdir(data_path) # List containing all the trajectories of the dataset

    lista = []

    histogram_data_x = [0 for i in range(256)] # Histogram data for x
    histogram_data_y = [0 for i in range(256)] # Histogram data for y
    histogram_data_z = [0 for i in range(256)] # Histogram data for z
    histogram_data_roll = [0 for i in range(256)] # Histogram data for roll
    histogram_data_pitch = [0 for i in range(256)] # Histogram data for pitch
    histogram_data_yaw = [0 for i in range(256)] # Histogram data for yaw

    if original:
        histogram_data_gripper = [0 for i in range(256)] # Continous gripper closedness
    else:
        histogram_data_gripper = [0 for i in range(2)] # Discrete gripper closedness

    j = 0

    for traj in traj_list:

        traj_path = os.path.join(data_path, traj) # Trajectory path
        # data = torch.load(traj_path) # Load the trajectory data
        with (open(traj_path, 'rb')) as f:
            data = pickle.load(f)

        episode = data["steps"] # Get the episode data

        for obs_dict in episode:

            action_x = torch.from_numpy(np.array(obs_dict["action"]["world_vector"][0]))
            action_y = torch.from_numpy(np.array(obs_dict["action"]["world_vector"][1]))
            action_z = torch.from_numpy(np.array(obs_dict["action"]["world_vector"][2]))
            action_roll = torch.from_numpy(np.array(obs_dict["action"]["rotation_delta"][0]))
            action_pitch = torch.from_numpy(np.array(obs_dict["action"]["rotation_delta"][1]))
            action_yaw = torch.from_numpy(np.array(obs_dict["action"]["rotation_delta"][2]))
            action_gripper_closedness = torch.from_numpy(obs_dict["action"]["gripper_closedness_action"])

            action = {
                'x_axis': action_x,
                'y_axis': action_y,
                'z_axis': action_z,
                'roll': action_roll,
                'pitch': action_pitch,
                'yaw': action_yaw,
                'gripper_closedness_action': action_gripper_closedness
            } 
            action_tokens = action_tokenizer.tokenize(action) # Tokenize the action
            action_detokenized = action_tokenizer.detokenize(action_tokens) # Detokenize the action
            

            histogram_data_x[action_tokens[0]] += 1 # Increment the histogram data for x
            histogram_data_y[action_tokens[1]] += 1 # Increment the histogram data for y
            histogram_data_z[action_tokens[2]] += 1 # Increment the histogram data for z
            histogram_data_roll[action_tokens[3]] += 1 # Increment the histogram data for roll
            histogram_data_pitch[action_tokens[4]] += 1 # Increment the histogram data for pitch
            histogram_data_yaw[action_tokens[5]] += 1 # Increment the histogram data for yaw
            if original:
                histogram_data_gripper[action_tokens[6]] += 1 # Increment the histogram data for gripper closedness
            else:
                histogram_data_gripper[action_tokens[6]] += 1 # Increment the histogram data for gripper closedness

    # Plot and save the histogram data

    build_histogram_and_save(histogram_data_x, original, changed_action_space, 'x')
    build_histogram_and_save(histogram_data_y, original, changed_action_space, 'y')
    build_histogram_and_save(histogram_data_z, original, changed_action_space, 'z')
    build_histogram_and_save(histogram_data_roll, original, changed_action_space, 'roll')
    build_histogram_and_save(histogram_data_pitch, original, changed_action_space, 'pitch')
    build_histogram_and_save(histogram_data_yaw, original, changed_action_space, 'yaw')
    build_histogram_and_save(histogram_data_gripper, original, changed_action_space, 'gripper_closedness')