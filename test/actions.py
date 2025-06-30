import os
import pickle
import torchvision.transforms as transforms

if __name__ == "__main__":

    data_path = "/mnt/localstorage/mspremulli/Datasets/Simulated_Converted_Delta2/"
    list_traj = os.listdir(data_path)

    for traj in list_traj:
        print("Current trajectory: ", traj)
        traj_path = data_path + traj
        with open(traj_path, "rb") as f:
            data = pickle.load(f)
        
        episode = data['steps']
        trajectory_length = len(episode)
        print("Trajectory length: ", trajectory_length)
        timestep = 0

        for obs in episode:
            
            action_world_vector = obs['action']['world_vector']
            action_gripper = obs['action']['gripper_closedness_action']
            if action_gripper[0] == 1:
                image = obs['observation']['image']
                image = image.squeeze()
                pilImg = transforms.ToPILImage()(image)
                pilImg.save("image_timestep_{}.png".format(timestep))
                print("Timestep {} Action Gripper {}: Action World Vector : {}".format(timestep, action_gripper, action_world_vector))
            timestep += 1
        
        print(BLOCCOSASA)


