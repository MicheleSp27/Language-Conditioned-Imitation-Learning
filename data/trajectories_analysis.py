# Script used to analyze the trajectories lengths of a given dataset.
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    
    data_path = "/mnt/localstorage/mspremulli/Datasets/RT-1_Converted_Dataset/" # Path to the dataset folder.
    original = True # If true, the original RT-1 dataset is analyzed. If false, MIVIA simulated dataset is analyzed.
    discard_length = 12 # Trajectories with a temporal length inferior to this value are loaded into a separate dictionary.

    
    traj_dict_short = {} # Dictionary storing as key the trajectory lenght and as value the number of trajectories with the given lenght.
    traj_dict_normal = {} # Dictionary storing as key the trajectory lenght and as value the number of trajectories with the given lenght.
    short_trajs = 0 # Counter of short trajectories.
    normal_trajs = 0 # Counter of normal trajectories.

    if not original:
        traj_data_len = len(os.listdir(data_path)) * 100 # Each task variation has 100 trajectories.
    else:
        traj_data_len = len(os.listdir(data_path)) # Size of the original dataset.
    
    # Accessing to the trajectories of the dataset
    for traj_index in tqdm(range(0, traj_data_len)):

        if not original:
            var_id = int(traj_index / 100) # Retrieving the variation id.
            traj_id = traj_index % 100 # Trajectory id inside the variation folder.
            if traj_id < 10:
                traj_id_str = "traj00{}.pkl".format(traj_id)
            elif traj_id < 100:
                traj_id_str = "traj0{}.pkl".format(traj_id)
            traj_path = os.path.join(data_path, "task_{}".format(var_id), traj_id_str) # Path to the trajectory file.
        else:
            traj_path = os.path.join(data_path, "traj{}.pkl".format(traj_index)) # Path to the trajectory file.

        # Loading trajectory data.
        with open(traj_path, "rb") as f:
            data = pickle.load(f)
        
        episode = data["steps"] # Trajectory data.
        episode_length = len(episode) # Length of the trajectory.

        # Update traj_dict.
        if episode_length <= discard_length:
            short_trajs += 1
            if traj_dict_short.get(episode_length, None) == None:
                traj_dict_short[episode_length] = 1
            else:
                traj_dict_short[episode_length] += 1
        
        normal_trajs += 1
        if traj_dict_normal.get(episode_length, None) == None:
            traj_dict_normal[episode_length] = 1
        else:
            traj_dict_normal[episode_length] += 1
        
    print("Number of trajectories with a lenght smaller than {} : {}".format(discard_length, short_trajs))
    print("Number of trajectories with a normal length : {}".format(normal_trajs))
    
    # Create plot.
    if original :
        file_name = "trajectory_lengts_short_RT_1_Dataset" if original else "trajectories_length_MIVIA_Dataset"
        plt.figure(figsize=(24, 20))
        plt.title("Distribution of trajectories with temporal length < {} steps".format(discard_length), fontsize = 30)
        plt.xticks(rotation = 90, fontsize = 26)
        plt.yticks(fontsize=26)
        plt.xlabel("Trajectories length", fontsize = 26)
        plt.ylabel("Trajectory count", fontsize = 26)
        plt.bar(list(traj_dict_short.keys()), traj_dict_short.values(), color='g', label = traj_dict_short.values())
        plt.savefig(file_name + ".pdf")

    # If original is false, all trajectories lenghts info are loaded in traj_dict_normal,
    file_name = "trajectory_lengts_normal_RT_1_Dataset" if original else "trajectories_length_MIVIA_Dataset"
    plt.title("Distribution of trajectories temporal length")
    plt.figure(figsize=(24, 20))
    plt.xticks(rotation = 90, fontsize = 26)
    plt.yticks(fontsize=26)
    plt.xlabel("Trajectories length")
    plt.ylabel("Trajectory count")
    plt.bar(list(traj_dict_normal.keys()), traj_dict_normal.values(), color='g', label = traj_dict_normal.values())
    plt.savefig(file_name + ".pdf")



    


        



