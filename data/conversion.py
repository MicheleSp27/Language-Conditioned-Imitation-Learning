# Script used to convert data into a suitable form for the dataset class.abs


import tensorflow as tf # Tensorflow Libraries are used to read the rt-1 original dataset
import tensorflow_datasets as tfds
import tensorflow_hub as hub # Used to Load the universal sentence encoder for natural language processing
import torch
from torchvision.transforms import ToTensor # Tensor conversion Function
from torchvision.transforms import InterpolationMode, functional # Resize function
import hydra 
import numpy as np
import os
import pickle
from torchvision.io import read_image
import torchvision.transforms as transforms
import sys
from torchvision.transforms.functional import resized_crop
import robosuite.utils.transform_utils as T
sys.path.insert(0,'/user/mspremulli/Language-Conditioned-Imitation-Learning/')
sys.path.insert(0,'/user/mspremulli/Multi-Task-LFD-Training-Framework/')

from training.multi_task_il.datasets.utils import create_data_aug

tf.config.set_visible_devices([], 'GPU')

@hydra.main(version_base=None, config_path="", config_name="conversion_config")
def dataset_conversion(config):

    # Reading Parameteres
    data_path = config.conversion_parameters.data_path # Path to the original data
    save_path = config.conversion_parameters.save_path # Path to the new converted data
    resize = config.conversion_parameters.resize # Boolean value to indicate whether perform resize or not
    if resize : # If resize is requested, set the new image height and width resolution
      image_height = config.conversion_parameters.image_height # New image height resolution
      image_width = config.conversion_parameters.image_width # New image width resolution
    convert_original = config.conversion_parameters.convert_original # Boolean value to indicate whether to convert the original dataset or MIVIA dataset
    if convert_original == False :
      simulated = config.conversion_parameters.simulated
    to_tensor = ToTensor() # Used for Tensor conversion. It automatically change channels to C, H, W in range [0,1]

    
    if convert_original :

      # RT-1 Dataset Conversion to pkl data.

      #Data Reading with tfds libraries
      rt_1 = tfds.builder_from_directory(data_path)
      info = rt_1.info
      rt_1.download_and_prepare()
      dataset = rt_1.as_data_source()["train"]

      # Data Conversion. Data is converted to the same original format for simplicity. 
      traj_index = 0
      for data in dataset:

          episode = data["steps"]
          traj_dict = {"steps" : []} # Trajectory dictionary containing as value a list of observation dictionary
          print("Episode length : {}".format(len(episode)))

          for j in range(len(episode)):

              obs_dict = episode[j]
              image = obs_dict["observation"]["image"]
              image = to_tensor(image) # C, H, W in range [0,1]
              if resize : 
                image = functional.resize(image,[image_height,image_width], interpolation = InterpolationMode.BILINEAR)
              obs_dict["observation"]["image"] = image
              traj_dict["steps"].append(obs_dict)
              
          torch.save(traj_dict, save_path + "traj{}".format(traj_index))
          traj_index = traj_index + 1
      
      print("The total number of trajectories is : {}".format(traj_index))

    # Conversion performed on the MIVIA Dataset
    else :
      
      embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/large/2") # Loading the universal sentence encoder 
      traj_index = 0 # Index of the trajectory
      list_dir = os.listdir(data_path) # List of the directories in the dataset
      uncorrect_dir = ['img','info.json','video','real_new_ur5e_pick_place_converted_absolute_heatmap_limited.png'] # List of the directories that are not trajectories
      language_instruction_dict = {"task_00" : "Pick green box and place it into the first bin", "task_01" : "Pick green box and place it into the second bin", "task_02" : "Pick green box and place it into the third bin", "task_03" : "Pick green box and place it into the fourth bin", "task_04" : "Pick yellow box and place it into the first bin", "task_05" : "Pick yellow box and place it into the second bin", "task_06" : "Pick yellow box and place it into the third bin", "task_07" : "Pick yellow box and place it into the fourth bin", "task_08" : "Pick blue box and place it into the first bin", "task_09" : "Pick blue box and place it into the second bin", "task_10" : "Pick blue box and place it into the third bin", "task_11" : "Pick blue box and place it into the fourth bin", "task_12" : "Pick red box and place it into the first bin", "task_13" : "Pick red box and place it into the second bin", "task_14" : "Pick red box and place it into the third bin", "task_15" : "Pick red box and place it into the fourth bin"} # Language Instruction Dictionary
      #Extracting trajectories from the directories
      min_x = 0
      max_x = 0
      min_y = 0
      max_y = 0
      min_z = 0
      max_z = 0
      min_xr = 0
      max_xr = 0
      min_yr = 0
      max_yr = 0
      min_zr = 0
      max_zr = 0
      
      for dir in list_dir:
        
        if dir not in uncorrect_dir:

          print("Converting trajectories in dir {} ...".format(dir))

          language_instruction = language_instruction_dict[dir] # Natural Language Instruction
          natural_language_embedding = embed([language_instruction]) # Computing the embedding of the language instruction using the same language encoder of RT-1
          natural_language_embedding = natural_language_embedding.numpy() # Conversion from tensorflow.python.framework.ops.EagerTensor' to numpy as the original dataset
          natural_language_embedding = np.squeeze(natural_language_embedding, axis = 0) # From (1, 512, ) to (512, ) like the original dataset
          print("The associated natural language instruction is : {}".format(language_instruction))
          trajectory_path_dir = data_path + dir + '/'
          trajectory_dir = os.listdir(trajectory_path_dir)

          for trajectory_pkl in trajectory_dir:
            
            with (open(trajectory_path_dir + trajectory_pkl, 'rb')) as openfile:

                  data = pickle.load(openfile) # Loading the data from the pickle file

                  
                  traj_dict = {'steps' : []} # Trajectory dictionary containing as value a list of observation dictionary                 
                  trajectory_length = len(data['traj']) # Length of the trajectory
                  trajectory = data['traj'] # Trajectory Data
                  
                  
                  #Extracting observations data from the trajectory
                  
                  for i in range(trajectory_length):

                    
                    data = trajectory[i] # Data is acessed as index based collection

                    observation = data['obs'] # Observation data
                    
                    image = observation['camera_front_image'] # 200, 360, 3
                
                    
                    """
                    pilImg = transforms.ToPILImage()(image)
                    pilImg.save("before_cropping_image_timestep_{}.png".format(i))
                    """

                    

                    # Resize is not needed. It's already performed by the function resized_crop
                    """
                    image = to_tensor(image) # C, H, W in range [0,1]
                    if resize : 
                      image = functional.resize(image,[image_height,image_width], interpolation = InterpolationMode.BILINEAR)
                    """
                    
                    # Crop Paramaters computation for the best crop in proprietary settings
                    
                    if simulated :
                      crop_params = [20, 25, 80, 75] # Crop Parameters for simulation images
                    else :
                      crop_params = [0, 30, 120, 120] # Crop Parameters for real images
                    # Render: top, distance_bottom, left, distance_right
                    top, left = crop_params[0], crop_params[2]
                    img_height, img_width = image.shape[0], image.shape[1]
                    box_h, box_w = img_height - top - \
                      crop_params[1], img_width - left - crop_params[3]


                    cropped_image = resized_crop(transforms.ToPILImage()(image), top=top, left=left, height=box_h,
                               width=box_w, size=(image_height,image_width))
                    
                    

                    # cropped_image.save("after_cropping_image_timestep_{}.png".format(i))

                    cropped_image = to_tensor(cropped_image)

                    

                    #Test Action. Ask to Francesco which action to use
                    

                    
                    # Delta Computation

                    if i == trajectory_length - 1:
                       
                       delta_eef_pos = np.zeros(3)
                       delta_eef_axisangle = np.zeros(3)
                       gripper = np.array([0]) 

                    else :
                      
                      current_eef_pos = data['obs']['eef_pos']
                      current_eef_quat = data['obs']['eef_quat']

                      next_eef_pos = trajectory[i + 1]['obs']['eef_pos']
                      next_eef_quat = trajectory[i + 1]['obs']['eef_quat']

                      gripper = trajectory[i + 1]['action'][6]

                      if gripper == -1:
                        gripper = np.array([0])
                      else:
                        gripper = np.array([1])

                      delta_eef_pos = next_eef_pos - current_eef_pos

                      delta_eef_quat = T.quat_distance(next_eef_quat, current_eef_quat)
                      delta_eef_axisangle = T.quat2axisangle(delta_eef_quat)

                      print("Timestep {} :".format(i))
                      print("Delta eef pos : {}".format(delta_eef_pos))

                      """
                      print("Current eef quat Timestep {} : {}".format(i, current_eef_quat))
                      print("Next eef quat Timestep {} : {}".format(i + 1, next_eef_quat))
                      print("Delta eef quat Timestep {} : {}".format(i, delta_eef_quat))

                      quat = T.axisangle2quat(delta_eef_axisangle)

                      print("Recomputed quat Timestep {} : {}".format(i, quat))
                      next_quat = T.quat_multiply(quat, current_eef_quat)
                      print("Next quat Timestep {} : {}".format(i + 1, next_quat))
                      """

                      


                    obs_dict = {'observation' : {'image' : cropped_image, 'natural_language_embedding' : natural_language_embedding, 'natural_language_instruction' : language_instruction}, 'action' : {'gripper_closedness_action' : gripper, 'rotation_delta' : delta_eef_axisangle, 'world_vector' : delta_eef_pos}}
                    traj_dict['steps'].append(obs_dict)

                    
                    if delta_eef_pos[0] < min_x:
                      min_x = delta_eef_pos[0]
                    if delta_eef_pos[0] > max_x:
                      max_x = delta_eef_pos[0]
                    if delta_eef_pos[1] < min_y:
                      min_y = delta_eef_pos[1]
                    if delta_eef_pos[1] > max_y:
                      max_y = delta_eef_pos[1]
                    if delta_eef_pos[2] < min_z:
                      min_z = delta_eef_pos[2]
                    if delta_eef_pos[2] > max_z:
                      max_z = delta_eef_pos[2]
                    if delta_eef_axisangle[0] < min_xr:
                      min_xr = delta_eef_axisangle[0]
                    if delta_eef_axisangle[0] > max_xr:
                      max_xr = delta_eef_axisangle[0]
                    if delta_eef_axisangle[1] < min_yr:
                      min_yr = delta_eef_axisangle[1]
                    if delta_eef_axisangle[1] > max_yr:
                      max_yr = delta_eef_axisangle[1]
                    if delta_eef_axisangle[2] < min_zr:
                      min_zr = delta_eef_axisangle[2]
                    if delta_eef_axisangle[2] > max_zr:
                      max_zr = delta_eef_axisangle[2]


                    

                  print(aaa)
                  with open(save_path + 'traj{}.pkl'.format(traj_index), 'wb') as f:
                    pickle.dump(traj_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                  

                  traj_index = traj_index + 1
                  
          
          print("Number of converted trajectories : {}".format(traj_index))
      
      print("The total number of trajectories is : {}".format(traj_index))
      print("The minimum x is : {}".format(min_x))
      print("The maximum x is : {}".format(max_x))
      print("The minimum y is : {}".format(min_y))
      print("The maximum y is : {}".format(max_y))
      print("The minimum z is : {}".format(min_z))
      print("The maximum z is : {}".format(max_z))
      print("The minimum xr is : {}".format(min_xr))
      print("The maximum xr is : {}".format(max_xr))
      print("The minimum yr is : {}".format(min_yr))
      print("The maximum yr is : {}".format(max_yr))
      print("The minimum zr is : {}".format(min_zr))
      print("The maximum zr is : {}".format(max_zr))
                  
if __name__ == "__main__":

    dataset_conversion()

    

    
    




