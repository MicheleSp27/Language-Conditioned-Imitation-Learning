import os
import pickle


# Actions min computation
def action_min(action_list):
    min = 1000
    for action in action_list:
        if action < min:
            min = action
    return min

# Actions max computation
def action_max(action_list):
    max = -1000
    for action in action_list:
        if action > max:
            max = action
    return max

# Actions mean computation
def action_mean(action_list):
    mean = 0
    for action in action_list:
        mean += action
    mean /= len(action_list)
    return mean

# Actions variance computation
def action_variance(action_list):
    mean = action_mean(action_list)
    variance = 0
    for action in action_list:
        variance += (action - mean) ** 2
    variance /= len(action_list)
    return variance

# Actions std computation
def action_std(action_list):
    variance = action_variance(action_list)
    std = variance ** 0.5
    return std


if __name__ == "__main__":

    data_path = "/mnt/localstorage/mspremulli/Datasets/sim_ur5e_pick_place_shifted_converted_delta/" # Path to the data
    dir_list = os.listdir(data_path) # Content od data_path

    x_actions_list = [] # List to store the x actions
    y_actions_list = [] # List to store the y actions
    z_actions_list = [] # List to store the z actions
    roll_actions_list = [] # List to store the roll actions
    pitch_actions_list = [] # List to store the pitch actions
    yaw_actions_list = [] # List to store the yaw actions

    for dir in dir_list:

        # dir is task_(index) where index goes from 0 to 15

        print("Current task: {}".format(dir))
        traj_path = data_path + dir + '/' # Inside each dir there are many trajectories
        traj_list = os.listdir(traj_path) # List of trajectories

        for traj in traj_list:

            with open(traj_path + traj, 'rb') as openfile:

                data = pickle.load(openfile) # Load the trajectory data
                
                trajectory_length = data['len'] # Extracting the length of the trajectory
                trajectory_data = data['traj'] # Extracting the trajectory data

                for i in range(trajectory_length):

                    obs = trajectory_data[i]
                    action = obs['action']

                    x_actions_list.append(action[0])
                    y_actions_list.append(action[1])
                    z_actions_list.append(action[2])
                    roll_actions_list.append(action[3])
                    pitch_actions_list.append(action[4])
                    yaw_actions_list.append(action[5])




    print("X actions min: {}".format(action_min(x_actions_list)))
    print("X actions max: {}".format(action_max(x_actions_list)))
    print("X actions mean: {}".format(action_mean(x_actions_list)))
    print("X actions variance: {}".format(action_variance(x_actions_list)))
    print("X actions std: {}".format(action_std(x_actions_list)))
    print("Y actions min: {}".format(action_min(y_actions_list)))
    print("Y actions max: {}".format(action_max(y_actions_list)))
    print("Y actions mean: {}".format(action_mean(y_actions_list)))
    print("Y actions variance: {}".format(action_variance(y_actions_list)))
    print("Y actions std: {}".format(action_std(y_actions_list)))
    print("Z actions min: {}".format(action_min(z_actions_list)))
    print("Z actions max: {}".format(action_max(z_actions_list)))
    print("Z actions mean: {}".format(action_mean(z_actions_list)))
    print("Z actions variance: {}".format(action_variance(z_actions_list)))
    print("Z actions std: {}".format(action_std(z_actions_list)))
    print("Roll actions min: {}".format(action_min(roll_actions_list)))
    print("Roll actions max: {}".format(action_max(roll_actions_list)))
    print("Roll actions mean: {}".format(action_mean(roll_actions_list)))
    print("Roll actions variance: {}".format(action_variance(roll_actions_list)))
    print("Roll actions std: {}".format(action_std(roll_actions_list)))
    print("Pitch actions min: {}".format(action_min(pitch_actions_list)))
    print("Pitch actions max: {}".format(action_max(pitch_actions_list)))
    print("Pitch actions mean: {}".format(action_mean(pitch_actions_list)))
    print("Pitch actions variance: {}".format(action_variance(pitch_actions_list)))
    print("Pitch actions std: {}".format(action_std(pitch_actions_list)))
    print("Yaw actions min: {}".format(action_min(yaw_actions_list)))
    print("Yaw actions max: {}".format(action_max(yaw_actions_list)))
    print("Yaw actions mean: {}".format(action_mean(yaw_actions_list)))
    print("Yaw actions variance: {}".format(action_variance(yaw_actions_list)))
    print("Yaw actions std: {}".format(action_std(yaw_actions_list)))


