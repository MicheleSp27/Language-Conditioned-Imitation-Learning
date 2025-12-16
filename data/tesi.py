import pickle
from PIL import Image
from torchvision import transforms
import cv2
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resized_crop
from torchvision.transforms import ToTensor # Tensor conversion Function


if __name__ == "__main__":

    # Used to extract an image from a trajectory
    """
    data_path = "/mnt/localstorage/mspremulli/Datasets/RT-1_Converted_Dataset/"
    traj_index = 3
    traj_path = data_path + "traj{}.pkl".format(traj_index)

    with open(traj_path, "rb") as f:
        data = pickle.load(f)

    episode = data["steps"]

    observation_image = episode[20]["observation"]["image"]
    pilImg = transforms.ToPILImage()(observation_image)
    pilImg.save("robotic_arm.png")
    """
    """
    traj_counter = 0
    data_path = "/mnt/localstorage/mspremulli/Datasets/RT-1_Converted_Dataset/"
    traj_index = 0


    while traj_counter < 50:

        traj_path = data_path + "traj{}.pkl".format(traj_index)

        with open(traj_path, "rb") as f:
            data = pickle.load(f)
    
        episode = data["steps"]
        language_instruction = episode[0]["observation"]["natural_language_instruction"]

        if "orange" in language_instruction.decode("utf-8"):

            traj_counter += 1

            video_name = "/user/mspremulli/Language-Conditioned-Imitation-Learning/data/video_orange/{}{}.mp4".format(language_instruction.decode("utf-8"), traj_counter)
            video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 5, (200,200))
            print("At traj_index {} the language instruction is : {}".format(traj_index,language_instruction))

            for obs_dict in episode:
                image = obs_dict["observation"]["image"]
                image = image.numpy()
                image = (image * 255).clip(0, 255).astype('uint8')
                image = image.transpose(1, 2, 0)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                video.write(image)
            
            video.release()
        
        traj_index += 1
    """
    
    
    """
    data_path = "/mnt/localstorage/mspremulli/Datasets/sim_ur5e_converted/task_15/"
    traj_index = 3
    traj_path = data_path + "traj00{}.pkl".format(traj_index)

    with open(traj_path, "rb") as f:
        data = pickle.load(f)
    
    episode = data["steps"]
    video_name = "/user/mspremulli/Language-Conditioned-Imitation-Learning/data/video_traj{}.avi".format(traj_index)
    video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 5, (200,200))
    language_instruction = episode[0]["observation"]["natural_language_instruction"]
    print("At traj_index {} the language instruction is : {}".format(traj_index,language_instruction))

    for obs_dict in episode:
        image = obs_dict["observation"]["image"]
        image = image.numpy()
        image = (image * 255).clip(0, 255).astype('uint8')
        image = image.transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)
    
    video.release()
    """
    """
    data_path = "/mnt/localstorage/mspremulli/Datasets/RT-1_Converted_Dataset/"
    j = 0

    for traj_index in range(0, len(os.listdir(data_path))):

        traj_path = data_path + "traj{}.pkl".format(traj_index)
        with open(traj_path, "rb") as f:
            data = pickle.load(f)
        
        episode = data["steps"]

        if j == 10:
            break

        if len(episode) > 12:
            continue
        
        j += 1
        video_name = "/user/mspremulli/Language-Conditioned-Imitation-Learning/data/video_traj{}_short.avi".format(traj_index)
        video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 2, (200,200))
        language_instruction = episode[0]["observation"]["natural_language_instruction"]
        print("At traj_index {} the language instruction is : {}".format(traj_index,language_instruction))

        for obs_dict in episode:
            image = obs_dict["observation"]["image"]
            image = image.numpy()
            image = (image * 255).clip(0, 255).astype('uint8')
            image = image.transpose(1, 2, 0)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image)
        
        video.release()
    """

    """
    labels = ["Reached Rate", "Picked Rate", "Success Rate"]
    means = [0.195, 0.041, 0.041]
    stds = [0.019, 0.014, 0.014]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel("Metric Value")
    plt.title("Performance Metrics on orange and grey boxes.")
    plt.tight_layout()
    plt.savefig("/user/mspremulli/Language-Conditioned-Imitation-Learning/data/metrics_orange_grey.pdf")
    """
    """
    data_path = "/mnt/localstorage/mspremulli/Datasets/RT-1_Converted_Dataset/"
    # traj_index = 46
    for traj_index in range(0, len(os.listdir(data_path))):

        traj_path = data_path + "traj{}.pkl".format(traj_index)

        with open(traj_path, "rb") as f:
            data = pickle.load(f)
    
        episode = data["steps"]
        video_name = "/user/mspremulli/Language-Conditioned-Imitation-Learning/data/video_knock.avi".format(traj_index)
        video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 5, (200,200))
        language_instruction = episode[0]["observation"]["natural_language_instruction"].decode("utf-8")
        if "knock" in language_instruction:
            print("At traj_index {} the language instruction is : {}".format(traj_index,language_instruction))

            for obs_dict in episode:
                image = obs_dict["observation"]["image"]
                image = image.numpy()
                image = (image * 255).clip(0, 255).astype('uint8')
                image = image.transpose(1, 2, 0)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                video.write(image)
            
            video.release()
            print(ahhh)
    """

    data_path = "/mnt/localstorage/mspremulli/Datasets/sim_ur5e_pick_place/task_00/traj000.pkl"
    to_tensor = ToTensor()

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    data = data["traj"]

    image = data[0]["obs"]["camera_front_image"]
    pilImg = transforms.ToPILImage()(image)
    pilImg.save("before_crop.png")
    crop_params = [20, 25, 80, 75] # Crop Parameters for simulation images

    # Render: top, distance_bottom, left, distance_right
    top, left = crop_params[0], crop_params[2]
    img_height, img_width = image.shape[0], image.shape[1]
    box_h, box_w = img_height - top - \
        crop_params[1], img_width - left - crop_params[3]

    cropped_image = resized_crop(transforms.ToPILImage()(image), top=top, left=left, height=box_h,width=box_w, size=(img_height,img_width))
    cropped_image = to_tensor(cropped_image)
    pilImg = transforms.ToPILImage()(cropped_image)
    pilImg.save("after_crop.png")




