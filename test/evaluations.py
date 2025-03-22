import sys
sys.path.insert(0,'/user/mspremulli/Language-Conditioned-IL/')

import hydra
import os
import torch
from torch.utils.data import RandomSampler
from model.transformer_network import TransformerNetwork
import time
from model.utils import batched_space_sampler, np_to_tensor
from model.action_tokenizer_cpu import RT1ActionTokenizer


@hydra.main(version_base=None, config_path="", config_name="evaluations_config")
def evaluations(config):
  
  data_path = config.evaluations_parameters.data_path # Path to the data
  checkpoint_path = config.evaluations_parameters.checkpoint_path # Checkpoint path
  number_of_trajectories = config.evaluations_parameters.number_of_trajectories # Number of trajectories to evaluate
  
  # Sampling Trajectories
  trajectory_list = os.listdir(data_path) 
  trajectory_sampler = RandomSampler(trajectory_list, replacement = False)
  trajectory_sampler_iterable = iter(trajectory_sampler)

  sampled_trajectories = []
  for i in range(number_of_trajectories):
    trajectory_index = next(trajectory_sampler_iterable)
    sampled_trajectories.append(trajectory_list[trajectory_index])

  # Load the model
  original = config.Original
  if original:
        print("Loading Original RT-1 Model...")
        model = hydra.utils.instantiate(config.Transformer_Original)
        print("Model Loaded")
  else:
        print("Loading MIVIA RT-1 Model...")
        model = hydra.utils.instantiate(config.Transformer_MIVIA)
        language_instruction_dict = {"task_00" : "Pick green box and place it into the first bin", "task_01" : "Pick green box and place it into the second bin", "task_02" : "Pick green box and place it into the third bin", "task_03" : "Pick green box and place it into the fourth bin", "task_04" : "Pick yellow box and place it into the first bin", "task_05" : "Pick yellow box and place it into the second bin", "task_06" : "Pick yellow box and place it into the third bin", "task_07" : "Pick yellow box and place it into the fourth bin", "task_08" : "Pick blue box and place it into the first bin", "task_09" : "Pick blue box and place it into the second bin", "task_10" : "Pick blue box and place it into the third bin", "task_11" : "Pick blue box and place it into the fourth bin", "task_12" : "Pick red box and place it into the first bin", "task_13" : "Pick red box and place it into the second bin", "task_14" : "Pick red box and place it into the third bin", "task_15" : "Pick red box and place it into the fourth bin"} # Language Instruction Dictionary
        print("Model Loaded")

  print("Loading Checkpoint {} ...".format(checkpoint_path))
  checkpoint = torch.load(checkpoint_path)
  print("Checkpoint Loaded")
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval() # Set the model to evaluation mode

  # Network State Variable Initialization

  network_state = batched_space_sampler(model._state_space, batch_size = 1)
  network_state = np_to_tensor(network_state)
  network_state["seq_idx"] = torch.tensor([0])

  # Action Tokenizer Initialization
  
  action_tokenizer = RT1ActionTokenizer(model._output_tensor_space, 256)

  # Evaluation Metrics Initialization
  accuracy_x = 0
  accuracy_y = 0
  accuracy_z = 0
  accuracy_rotation_x = 0
  accuracy_rotation_y = 0
  accuracy_rotation_z = 0
  accuracy_gripper = 0
  number_of_observation = 0



  for traj_path in sampled_trajectories:
      
      trajectory = torch.load(data_path + traj_path)
      trajectory_data = trajectory["steps"]
      trajectory_length = len(trajectory_data)

      accuracy_x_traj = 0
      accuracy_y_traj = 0
      accuracy_z_traj = 0
      accuracy_rotation_x_traj = 0
      accuracy_rotation_y_traj = 0
      accuracy_rotation_z_traj = 0
      accuracy_gripper_traj = 0

      natural_language_instruction = trajectory_data[0]["observation"]["natural_language_instruction"]

         
      print("Task : {}".format(natural_language_instruction))
      print("Length : {}".format(trajectory_length))

      for obs_index in range(trajectory_length):
          
          # Extracting the observation data
          obs_dict = trajectory_data[obs_index]
          image = obs_dict["observation"]["image"]
          image = image.unsqueeze(0)
          natural_language_embedding = obs_dict["observation"]["natural_language_embedding"]
          natural_language_embedding = torch.from_numpy(natural_language_embedding)
          action_world_vector = torch.from_numpy(obs_dict["action"]["world_vector"])
          action_rotation_delta = torch.from_numpy(obs_dict["action"]["rotation_delta"])
          action_gripper_closedness = torch.from_numpy(obs_dict["action"]["gripper_closedness_action"])

          # Ground Truth Action
          action = {"world_vector": action_world_vector, "rotation_delta": action_rotation_delta, "gripper_closedness_action": action_gripper_closedness}
          action_tokens = action_tokenizer.tokenize(action)

          # Model Prediction
          observation = {"image": image, "natural_language_embedding": natural_language_embedding}
          predicted_action, network_state = model(observation, network_state)

          # Extracting the predicted action tokens
          if obs_index <= 5:
            predicted_action_tokens = network_state["action_tokens"][0][obs_index]
          else :
            predicted_action_tokens = network_state["action_tokens"][0][5]
          
          # Comparing the predicted action tokens with the ground truth action tokens

            if action_tokens[0] == predicted_action_tokens[0]:
              accuracy_x_traj += 1
              accuracy_x += 1
            if action_tokens[1] == predicted_action_tokens[1]:
              accuracy_y_traj += 1
              accuracy_y += 1
            if action_tokens[2] == predicted_action_tokens[2]:
              accuracy_z_traj += 1
              accuracy_z += 1
            if action_tokens[3] == predicted_action_tokens[3]:
              accuracy_rotation_x_traj += 1
              accuracy_rotation_x += 1
            if action_tokens[4] == predicted_action_tokens[4]:
              accuracy_rotation_y_traj += 1
              accuracy_rotation_y += 1
            if action_tokens[5] == predicted_action_tokens[5]:
              accuracy_rotation_z_traj += 1
              accuracy_rotation_z += 1
            if action_tokens[6] == predicted_action_tokens[6]:
              accuracy_gripper_traj += 1
              accuracy_gripper += 1
            
            number_of_observation += 1
          
      print("Accuracy X : {}".format(accuracy_x_traj/trajectory_length))
      print("Accuracy Y : {}".format(accuracy_y_traj/trajectory_length))
      print("Accuracy Z : {}".format(accuracy_z_traj/trajectory_length))
      print("Accuracy Rotation X : {}".format(accuracy_rotation_x_traj/trajectory_length))
      print("Accuracy Rotation Y : {}".format(accuracy_rotation_y_traj/trajectory_length))
      print("Accuracy Rotation Z : {}".format(accuracy_rotation_z_traj/trajectory_length))
      print("Accuracy Gripper : {}".format(accuracy_gripper_traj/trajectory_length))

  print("Overall Accuracy X : {}".format(accuracy_x/number_of_observation))
  print("Overall Accuracy Y : {}".format(accuracy_y/number_of_observation))
  print("Overall Accuracy Z : {}".format(accuracy_z/number_of_observation))
  print("Overall Accuracy Rotation X : {}".format(accuracy_rotation_x/number_of_observation))
  print("Overall Accuracy Rotation Y : {}".format(accuracy_rotation_y/number_of_observation))
  print("Overall Accuracy Rotation Z : {}".format(accuracy_rotation_z/number_of_observation))
  print("Overall Accuracy Gripper : {}".format(accuracy_gripper/number_of_observation))

         

          



if __name__ == "__main__":
  evaluations()