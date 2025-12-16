import sys
sys.path.insert(0,'/user/mspremulli/Language-Conditioned-Imitation-Learning/')

import hydra
import os
import torch
from torch.utils.data import RandomSampler
from model.transformer_network import TransformerNetwork
import time
import pickle
from model.utils import batched_space_sampler, np_to_tensor
from model.action_tokenizer import RT1ActionTokenizer
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm


@hydra.main(version_base=None, config_path="", config_name="evaluations_config")
def evaluations(config):

  # Setting GPU or CPU. GPU is preferred for inference.
  if torch.cuda.is_available():
      print("Using GPU {} for testing.".format(torch.cuda.get_device_name(0)))
      device = torch.device("cuda")
  else:
      print("Using CPU for testing.")
      device = torch.device("cpu")
  
  # Reading the configuration parameters from the config file.
  original = config.evaluations_parameters.original # True if working with the original robotic platform and dataset. False, for MIVIA robotic platform and dataset.
  data_path = config.evaluations_parameters.data_path # Path to the data.
  checkpoint_path = config.evaluations_parameters.checkpoint_path # Path to the model checkpoint to be tested. 
  number_of_trajectories = config.evaluations_parameters.number_of_trajectories # Number of trajectories to evaluate. Used only if working with balanced set to False or testing on the real RT-1 Dataset.
  interval = config.evaluations_parameters.interval # Accuracy over the specified action token interval.
  balanced = config.evaluations_parameters.balanced # Balanced set test of trajectories. Valid only for the MIVIA Simulated dataset.
  number_of_trajectories_for_variation = config.evaluations_parameters.number_of_trajectories_for_variation # Number of trajectories for task variations. Valid only for the MIVIA Simulated Dataset.
  seed = config.evaluations_parameters.seed # Seed for test reproducibility.
  generator = torch.Generator().manual_seed(seed) # Generator for sampling the trajectories for evaluation.
  printing = config.evaluations_parameters.printing # If true, prints evaluation results step by step. False, only final results are printed.
  

  # Sampling trajectories to evaluate on.
  print("Sampling trajectories for evaluation...")
  if original :
     number_of_trajectories_dataset = len(os.listdir(data_path))
  else:
     number_of_variations = 16
     number_of_trajectories_per_variation = 100
     number_of_trajectories_dataset = len(os.listdir(data_path)) * number_of_trajectories_per_variation # 16 total variation, each containing 100 trajectories.
  
  sampled_trajectories = [] # List containing the index of the sampled trajectories for evaluation.
  available_trajectories = [i for i in range(0, number_of_trajectories_dataset)] # List containing the index of the available trajectories for evaluation.

  if balanced == False:
     # RandomSampler definition.
     trajectory_sampler = RandomSampler(available_trajectories, replacement = False, generator = generator)
     trajectory_sampler_iterable = iter(trajectory_sampler)

     for _ in range(0, number_of_trajectories):
        
        # Sampling the trajectory at random.
        sampled_index_trajectory = next(trajectory_sampler_iterable) 
        sampled_trajectory = available_trajectories[sampled_index_trajectory]
        sampled_trajectories.append(sampled_trajectory)
  
  else:
     
     assert original == False, "Only available on the MIVIA Simulated Dataset!"

     for var_id in range(0, number_of_variations):
        
        # Definining the sampler over the trajectories associated to the specific task variation.
        variation_low_index = var_id * 100
        variation_high_index = variation_low_index + 100
        variation_sampler = RandomSampler(available_trajectories[variation_low_index:variation_high_index], replacement = False, generator = generator)
        variation_sampler_iterable = iter(variation_sampler)

        for _ in range(0, number_of_trajectories_for_variation):

          # Sampling the trajectory at random for the specific variation id.
          sampled_index_trajectory = next(variation_sampler_iterable) 
          sampled_trajectory = available_trajectories[sampled_index_trajectory]
          sampled_trajectories.append(sampled_trajectory)
  
  print("Sampled {} trajectories for evaluation.".format(len(sampled_trajectories)))

  # Loading RT-1 Transformer based model.
  if original:
        print("Loading RT-1 Model with original action space...")
        model = hydra.utils.instantiate(config.Transformer_Original).to(device)
        print("Model Loaded")
  else:
        print("Loading RT-1 Model with custom action space...")
        model = hydra.utils.instantiate(config.Transformer_MIVIA).to(device)
        print("Model Loaded")

  # Loading Checkpoint.
  print("Loading Checkpoint {} ...".format(checkpoint_path))
  checkpoint = torch.load(checkpoint_path, map_location=device)
  epoch = checkpoint["epoch"]
  model.load_state_dict(checkpoint["model_state_dict"])
  print("{} Epoch/s Checkpoint loaded.".format(epoch))
  model.eval() # Setting the model in inference mode.

  if original:
    file_result_path = "RT-1_Original_Epoch-{}_Interval-{}.txt".format(epoch, interval)
  else:
    if balanced:
      file_result_path = "RT-1_MIVIA_Balance_Epoch-{}_Interval-{}.txt".format(epoch, interval)
    else:
      file_result_path = "RT-1_MIVIA_Epoch-{}_Interval-{}.txt".format(epoch, interval)
  
  f = open(file_result_path, "w")
  f.write("Testing RT-1 on {} trajectories.\n".format(number_of_trajectories if balanced == False else number_of_trajectories_for_variation * number_of_variations))

  # Action Tokenizer Initialization. The action space is retrieved from the initialized model. The number of bin is set to 256 like the original work.
  action_tokenizer = RT1ActionTokenizer(model._output_tensor_space, 256)

  # Evaluation Metrics Initialization.
  accuracy_x = 0 # Accuracy over x axis action token.
  accuracy_y = 0 # Accuracy over y axis action token.
  accuracy_z = 0 # Accuracy over z axis action token.
  accuracy_roll = 0 # Accuracy over roll action token.
  accuracy_pitch = 0 # Accuracy over pitch action token.
  accuracy_yaw = 0 # Accuracy over yaw action token.
  accuracy_gripper = 0 # Accuracy over gripper action token.

  accuracy_x_interval = 0 # Accuracy over x axis action token within the specified interval.
  accuracy_y_interval = 0 # Accuracy over y axis action token within the specified interval.
  accuracy_z_interval = 0 # Accuracy over z axis action token within the specified interval.
  accuracy_roll_interval = 0 # Accuracy over roll action token within the specified interval.
  accuracy_pitch_interval = 0 # Accuracy over pitch action token within the specified interval.
  accuracy_yaw_interval = 0 # Accuracy over yaw action token within the specified interval.
  accuracy_gripper_interval = 0 # Accuracy over gripper action token within the specified interval.

  number_of_observation = 0 # Total number of observations evaluated.

  with torch.no_grad():

    for index in tqdm(range(0, len(sampled_trajectories))):
        
        traj_index = sampled_trajectories[index] # Retrieving the trajectory index.
        # At the start of each trajectory evaluation, the network state must be re-initialized. 
        network_state = batched_space_sampler(model._state_space, batch_size = 1)
        network_state = np_to_tensor(network_state)
        network_state["seq_idx"] = torch.tensor([0])
        network_state["action_tokens"] = network_state["action_tokens"].to(device)
        network_state["context_image_tokens"] = network_state["context_image_tokens"].to(device)
        network_state["seq_idx"] = network_state["seq_idx"].to(device)

        # Accuracy over a single trajectory.
        accuracy_x_traj = 0 # Accuracy over x axis action token.
        accuracy_y_traj = 0 # Accuracy over y axis action token.
        accuracy_z_traj = 0 # Accuracy over z axis action token.
        accuracy_roll_traj= 0 # Accuracy over roll action token.
        accuracy_pitch_traj = 0 # Accuracy over pitch action token.
        accuracy_yaw_traj = 0 # Accuracy over yaw action token.
        accuracy_gripper_traj = 0  # Accuracy over gripper action token.

        accuracy_x_traj_interval = 0 # Accuracy over x axis action token within the specified interval.
        accuracy_y_traj_interval = 0 # Accuracy over y axis action token within the specified interval.
        accuracy_z_traj_interval = 0 # Accuracy over z axis action token within the specified interval.
        accuracy_roll_traj_interval = 0 # Accuracy over roll action token within the specified interval.
        accuracy_pitch_traj_interval = 0 # Accuracy over pitch action token within the specified interval.
        accuracy_yaw_traj_interval = 0 # Accuracy over yaw action token within the specified interval.
        accuracy_gripper_traj_interval = 0 # Accuracy over gripper action token within the specified interval.

        # Loading trajectory data.
        if original:
          traj_path = os.path.join(data_path, "traj{}.pkl".format(traj_index))
        else:
          variation_id = int(traj_index / 100)
          trajectory_id = traj_index % 100
          if trajectory_id < 10:
            traj_id = "traj00{}.pkl".format(trajectory_id)
          else:
            traj_id = "traj0{}.pkl".format(trajectory_id)
          traj_path = os.path.join(data_path, "task_{}".format(variation_id), traj_id )

        with open(traj_path, "rb") as file:
            trajectory = pickle.load(file)
        trajectory_data = trajectory["steps"] # Retrieving the trajectory steps data.

        trajectory_length = len(trajectory_data)
        natural_language_instruction = trajectory_data[0]["observation"]["natural_language_instruction"]
        
        f.write("Task : {} \n".format(natural_language_instruction))
        f.write("Task episode length : {} \n".format(trajectory_length))

        if printing:

          print("Task : {}".format(natural_language_instruction))
          print("Task episode length : {}".format(trajectory_length))

        for obs_index in range(trajectory_length):
            
            number_of_observation += 1 # Updating the total number of observations evaluated.

            obs_dict = trajectory_data[obs_index] # Retrieving the observation dictionary at the specific timestep.
            image = obs_dict["observation"]["image"] # Retrieving the image observation.
            image = image.unsqueeze(0) # Adding an additional batch dimension.
            natural_language_embedding = obs_dict["observation"]["natural_language_embedding"] # Retrieving the natural language embedding observation.
            natural_language_embedding = torch.from_numpy(natural_language_embedding) # Converting the natural language embedding to a torch tensor.

            if original :
              # If working on the original RT-1 dataset, then actions are specified in this format.
              action_translation = torch.from_numpy(np.array(obs_dict["action"]["world_vector"])).to(device) # Retrieving the translation delta observation.
              action_rotation = torch.from_numpy(np.array(obs_dict["action"]["rotation_delta"])).to(device) # Retrieving the rotation delta observation.
              action_gripper_closedness = torch.from_numpy(np.array(obs_dict["action"]["gripper_closedness_action"])).to(device) # Retrieving the gripper closedness observation.
              action = {"world_vector" : action_translation, "rotation_delta" : action_rotation, "gripper_closedness_action" : action_gripper_closedness}
            else : 
              # If working on the MIVIA RT-1 dataset, then actions are specified in this format.
              action_x = torch.from_numpy(np.array(obs_dict["action"]["world_vector"][0])) # Retrieving the ground truth action x axis.
              action_y = torch.from_numpy(np.array(obs_dict["action"]["world_vector"][1])) # Retrieving the ground truth action y axis.
              action_z = torch.from_numpy(np.array(obs_dict["action"]["world_vector"][2])) # Retrieving the ground truth action z axis.
              action_roll = torch.from_numpy(np.array(obs_dict["action"]["rotation_delta"][0])) # Retrieving the ground truth action roll.
              action_pitch = torch.from_numpy(np.array(obs_dict["action"]["rotation_delta"][1])) # Retrieving the ground truth action pitch.
              action_yaw = torch.from_numpy(np.array(obs_dict["action"]["rotation_delta"][2])) # Retrieving the ground truth action yaw.
              action_gripper_closedness = torch.from_numpy(obs_dict["action"]["gripper_closedness_action"]) # Retrieving the ground truth action gripper closedness.
              action = {'x_axis': action_x.to(device), 'y_axis': action_y.to(device), 'z_axis': action_z.to(device), 'roll': action_roll.to(device), 'pitch': action_pitch.to(device), 'yaw': action_yaw.to(device), 'gripper_closedness_action': action_gripper_closedness.to(device)}
              
            # Tokenizing the ground truth action.
            action_tokens = action_tokenizer.tokenize(action) # Tokenizing the ground truth action.
            
            # Inference step.
            observation = {"image": image.to(device), "natural_language_embedding": natural_language_embedding.to(device)}
            predicted_action, network_state = model(observation, network_state)
          
            # Extracting the predicted action tokens from the network state.
            if obs_index < 5:
              predicted_action_tokens = network_state["action_tokens"][0][obs_index]
            else :
              predicted_action_tokens = network_state["action_tokens"][0][5]

            # Exact Match Evaluation.
            if action_tokens[0] == predicted_action_tokens[0]: # x axis
              accuracy_x_traj += 1
              accuracy_x += 1
            if action_tokens[1] == predicted_action_tokens[1]: # y axis
              accuracy_y_traj += 1
              accuracy_y += 1
            if action_tokens[2] == predicted_action_tokens[2]: # z axis
              accuracy_z_traj += 1
              accuracy_z += 1
            if action_tokens[3] == predicted_action_tokens[3]: # roll
              accuracy_roll_traj+= 1
              accuracy_roll += 1
            if action_tokens[4] == predicted_action_tokens[4]: # pitch
              accuracy_pitch_traj += 1
              accuracy_pitch += 1
            if action_tokens[5] == predicted_action_tokens[5]: # yaw
              accuracy_yaw_traj += 1
              accuracy_yaw += 1
            if action_tokens[6] == predicted_action_tokens[6]: # gripper
              accuracy_gripper_traj += 1
              accuracy_gripper += 1

            # Interval Evaluation
            if abs(action_tokens[0] - predicted_action_tokens[0]) <= interval: # x axis
              accuracy_x_traj_interval += 1
              accuracy_x_interval += 1
            if abs(action_tokens[1] - predicted_action_tokens[1]) <= interval: # y axis
              accuracy_y_traj_interval += 1
              accuracy_y_interval += 1
            if abs(action_tokens[2] - predicted_action_tokens[2]) <= interval: # z axis
              accuracy_z_traj_interval += 1
              accuracy_z_interval += 1
            if abs(action_tokens[3] - predicted_action_tokens[3]) <= interval: # roll
              accuracy_roll_traj_interval += 1
              accuracy_roll_interval += 1
            if abs(action_tokens[4] - predicted_action_tokens[4]) <= interval: # pitch
              accuracy_pitch_traj_interval += 1
              accuracy_pitch_interval += 1
            if abs(action_tokens[5] - predicted_action_tokens[5]) <= interval: # yaw
              accuracy_yaw_traj_interval += 1
              accuracy_yaw_interval += 1
            if abs(action_tokens[6] - predicted_action_tokens[6]) <= interval: # gripper
              accuracy_gripper_traj_interval += 1
              accuracy_gripper_interval += 1
    
        if printing:
          # Printing the results
          print("Accuracy X : {}".format(accuracy_x_traj/trajectory_length)) 
          print("Accuracy Y : {}".format(accuracy_y_traj/trajectory_length))
          print("Accuracy Z : {}".format(accuracy_z_traj/trajectory_length))
          print("Accuracy Rotation X : {}".format(accuracy_roll_traj/trajectory_length))
          print("Accuracy Rotation Y : {}".format(accuracy_pitch_traj/trajectory_length))
          print("Accuracy Rotation Z : {}".format(accuracy_yaw_traj/trajectory_length))
          print("Accuracy Gripper : {}".format(accuracy_gripper_traj/trajectory_length))
        
        # Writing the trajectory results on the file.
        f.write("Accuracy X : {} \n".format(accuracy_x_traj/trajectory_length))
        f.write("Accuracy Y : {} \n".format(accuracy_y_traj/trajectory_length))
        f.write("Accuracy Z : {} \n".format(accuracy_z_traj/trajectory_length))
        f.write("Accuracy Roll : {} \n".format(accuracy_roll_traj/trajectory_length))
        f.write("Accuracy Pitch : {} \n".format(accuracy_pitch_traj/trajectory_length))
        f.write("Accuracy Yaw : {} \n".format(accuracy_yaw_traj/trajectory_length))
        f.write("Accuracy Gripper : {} \n".format(accuracy_gripper_traj/trajectory_length))
        
        if printing:
          # Printing interval accuracy.
          print("Interval Accuracy X : {}".format(accuracy_x_traj_interval/trajectory_length))
          print("Interval Accuracy Y : {}".format(accuracy_y_traj_interval/trajectory_length))
          print("Interval Accuracy Z : {}".format(accuracy_z_traj_interval/trajectory_length))
          print("Interval Accuracy Roll : {}".format(accuracy_roll_traj_interval/trajectory_length))
          print("Interval Accuracy Pitch : {}".format(accuracy_pitch_traj_interval/trajectory_length))
          print("Interval Accuracy Yaw : {}".format(accuracy_yaw_traj_interval/trajectory_length))
          print("Interval Accuracy Gripper : {}".format(accuracy_gripper_traj/trajectory_length))
      
        # Writing the trajectory interval results on the file.
        f.write("Interval Accuracy X : {} \n".format(accuracy_x_traj_interval/trajectory_length))
        f.write("Interval Accuracy Y : {} \n".format(accuracy_y_traj_interval/trajectory_length))
        f.write("Interval Accuracy Z : {} \n".format(accuracy_z_traj_interval/trajectory_length))
        f.write("Interval Accuracy Roll : {} \n".format(accuracy_roll_traj_interval/trajectory_length))
        f.write("Interval Accuracy Pitch : {} \n".format(accuracy_pitch_traj_interval/trajectory_length))
        f.write("Interval Accuracy Yaw : {} \n".format(accuracy_yaw_traj_interval/trajectory_length))
        f.write("Interval Accuracy Gripper : {} \n\n\n".format(accuracy_gripper_traj_interval/trajectory_length))

    print("Overall Accuracy X : {}".format(accuracy_x/number_of_observation))
    print("Overall Accuracy Y : {}".format(accuracy_y/number_of_observation))
    print("Overall Accuracy Z : {}".format(accuracy_z/number_of_observation))
    print("Overall Accuracy Rotation X : {}".format(accuracy_roll/number_of_observation))
    print("Overall Accuracy Rotation Y : {}".format(accuracy_pitch/number_of_observation))
    print("Overall Accuracy Rotation Z : {}".format(accuracy_yaw/number_of_observation))
    print("Overall Accuracy Gripper : {}".format(accuracy_gripper/number_of_observation))
    print("Overall Interval Accuracy X : {}".format(accuracy_x_interval/number_of_observation))
    print("Overall Interval Accuracy Y : {}".format(accuracy_y_interval/number_of_observation))
    print("Overall Interval Accuracy Z : {}".format(accuracy_z_interval/number_of_observation))
    print("Overall Interval Accuracy Roll : {}".format(accuracy_roll_interval/number_of_observation))
    print("Overall Interval Accuracy Pitch : {}".format(accuracy_pitch_interval/number_of_observation))
    print("Overall Interval Accuracy Yaw : {}".format(accuracy_yaw_interval/number_of_observation))
    
    f.write("Overall Accuracy X : {} \n".format(accuracy_x/number_of_observation))
    f.write("Overall Accuracy Y : {} \n".format(accuracy_y/number_of_observation))
    f.write("Overall Accuracy Z : {} \n".format(accuracy_z/number_of_observation))
    f.write("Overall Accuracy Roll : {} \n".format(accuracy_roll/number_of_observation))
    f.write("Overall Accuracy Pitch : {} \n".format(accuracy_pitch/number_of_observation))
    f.write("Overall Accuracy Yaw : {} \n".format(accuracy_yaw/number_of_observation))
    f.write("Overall Accuracy Gripper : {} \n".format(accuracy_gripper/number_of_observation))
    f.write("Overall Interval Accuracy X : {} \n".format(accuracy_x_interval/number_of_observation))
    f.write("Overall Interval Accuracy Y : {} \n".format(accuracy_y_interval/number_of_observation))
    f.write("Overall Interval Accuracy Z : {} \n".format(accuracy_z_interval/number_of_observation))
    f.write("Overall Interval Accuracy Roll : {} \n".format(accuracy_roll_interval/number_of_observation))
    f.write("Overall Interval Accuracy Pitch : {} \n".format(accuracy_pitch_interval/number_of_observation))
    f.write("Overall Interval Accuracy Yaw : {} \n".format(accuracy_yaw_interval/number_of_observation))
    f.write("Overall Interval Accuracy Gripper : {} \n".format(accuracy_gripper_interval/number_of_observation))
    f.close()

if __name__ == "__main__":
  evaluations()