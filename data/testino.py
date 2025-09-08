from torch.utils.data import Sampler, RandomSampler
import pickle
import cv2
import numpy as np
import torch

if __name__ == "__main__":

    data_path = "/mnt/localstorage/mspremulli/Datasets/Simulated_Converted_Delta2/"
    number_of_sampling = 30
    for i in range(16):

        low_bound = i * 100
        high_bound = low_bound + 100

        traj_index_list = [j for j in range(low_bound, high_bound)]
        sampler = RandomSampler(traj_index_list, replacement = False)
        iterable = iter(sampler)

        for j in range(number_of_sampling):

            sampled_index = next(iterable)
            sampled_trajectory = traj_index_list[sampled_index]

            traj_name = "traj{}.pkl".format(sampled_trajectory)

            traj_path = data_path + traj_name

            with(open(traj_path, "rb")) as f:

                data = pickle.load(f)

            episode = data["steps"]
            video_name = "/user/mspremulli/Language-Conditioned-Imitation-Learning/data/video/" + "traj{}.avi".format(sampled_trajectory)
            video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 5, (200,200))
            language_instruction = episode[0]["observation"]["natural_language_instruction"]
            print(language_instruction)
            for observation_index in range(len(episode)):

                image = episode[observation_index]["observation"]["image"]
                image = torch.permute(image, (1,2,0))
                image = image.numpy()
                image = (image * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert the image from RGB to BGR
                video.write(image) # Write the image to the video
                
            video.release()
        
        print("------------------------------------")




