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


@hydra.main(version_base=None, config_path="", config_name="evaluations_config")
def evaluations(config):

  if torch.cuda.is_available():
      print(torch.cuda.get_device_name(0))
      device = torch.device("cuda")
      print("Cuda")
  else:
      device = torch.device("cpu")
  
  data_path = config.evaluations_parameters.data_path # Path to the data
  checkpoint_path = config.evaluations_parameters.checkpoint_path # Checkpoint path
  number_of_trajectories = config.evaluations_parameters.number_of_trajectories # Number of trajectories to evaluate
  interval = config.evaluations_parameters.interval # Interval accuracy 
  checkpoint = config.evaluations_parameters.checkpoint # Use checkpoint or not
  balanced = config.evaluations_parameters.balanced # Use balanced dataset or not
  number_of_trajectories_for_variation = config.evaluations_parameters.number_of_trajectories_for_variation # Number of trajectories for variations
  changed_action_space = config.evaluations_parameters.changed_action_space # Use changed action space or not
  use_predefined_trajectories = config.evaluations_parameters.use_predefined_trajectories # Use predefined trajectories or not
  # Sampling Trajectories


  sampled_trajectories = []
  if use_predefined_trajectories == False:
    if balanced:
      
      print("Balanced Sampling...")
      trajectory_list = os.listdir(data_path)
      number_of_variations = int(len(trajectory_list)/100)
      assert number_of_variations == 16

      for i in range(number_of_variations):
          
          lower_bound = i*100
          upper_bound = lower_bound + 100
          sample_list = []

          for j in range(lower_bound, upper_bound):
            sample_list.append(j)

          trajectory_sampler = RandomSampler(sample_list, replacement = False)
          trajectory_sampler_iterable = iter(trajectory_sampler)

          for k in range(number_of_trajectories_for_variation):
            trajectory_index = next(trajectory_sampler_iterable)
            sampled_trajectory = "traj" + str(sample_list[trajectory_index])
            sampled_trajectories.append(sampled_trajectory)
            
    else :

      print("Random Sampling...")
      
      trajectory_list = os.listdir(data_path)
      trajectory_sampler = RandomSampler(trajectory_list, replacement = False)
      trajectory_sampler_iterable = iter(trajectory_sampler)

      for i in range(number_of_trajectories):
        trajectory_index = next(trajectory_sampler_iterable)
        sampled_trajectories.append(trajectory_list[trajectory_index])
  else :
    print("Predefined Set of Trajectories...")
    sampled_trajectories = torch.load("/user/mspremulli/Language-Conditioned-Imitation-Learning/test/sampled_trajectories.pkl")


  # Load the model
  original = config.Original
  if original:
        print("Loading Original RT-1 Model...")
        model = hydra.utils.instantiate(config.Transformer_Original).to(device)
        print("Model Loaded")
  else:
        print("Loading MIVIA RT-1 Model...")
        model = hydra.utils.instantiate(config.Transformer_MIVIA).to(device)
        language_instruction_dict = {"task_00" : "Pick green box and place it into the first bin", "task_01" : "Pick green box and place it into the second bin", "task_02" : "Pick green box and place it into the third bin", "task_03" : "Pick green box and place it into the fourth bin", "task_04" : "Pick yellow box and place it into the first bin", "task_05" : "Pick yellow box and place it into the second bin", "task_06" : "Pick yellow box and place it into the third bin", "task_07" : "Pick yellow box and place it into the fourth bin", "task_08" : "Pick blue box and place it into the first bin", "task_09" : "Pick blue box and place it into the second bin", "task_10" : "Pick blue box and place it into the third bin", "task_11" : "Pick blue box and place it into the fourth bin", "task_12" : "Pick red box and place it into the first bin", "task_13" : "Pick red box and place it into the second bin", "task_14" : "Pick red box and place it into the third bin", "task_15" : "Pick red box and place it into the fourth bin"} # Language Instruction Dictionary
        print("Model Loaded")



  if checkpoint:

    print("Loading Checkpoint {} ...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Checkpoint Loaded")

    if original:
      print("Testing RT-1 on the original dataset trained for {} epochs on {} trajectories with interval set to {}".format(epoch, number_of_trajectories, interval))
      file_result_path = "Results-RT-1-Original-Epoch-{}-Interval-{}-Traj-{}.txt".format(epoch,interval, number_of_trajectories)
      file = open(file_result_path, "w")
      file.write("Testing RT-1 on the original dataset trained for {} epochs on {} trajectories with interval set to {}".format(epoch, number_of_trajectories, interval))
      file.write("\n")
    else :
      if balanced == False and changed_action_space == False:
        print("Testing MIVIA RT-1 on the simulated dataset trained for {} epochs on {} trajectories with interval set to {}".format(epoch, number_of_trajectories, interval))
        file_result_path = "Results-MIVIA-RT-1-Epoch-{}-Interval-{}-Traj-{}.txt".format(epoch,interval, number_of_trajectories)
      elif balanced == True and changed_action_space == True:
        print("Testing MIVIA RT-1 on the simulated dataset trained for {} epochs on {} trajectories with balanced sampling with interval set to {}".format(epoch, number_of_trajectories, interval))
        file_result_path = "Results-MIVIA-RT-1-Balanced-ChangedActionSpace-Epoch-{}-Interval-{}-Traj-{}.txt".format(epoch, interval, number_of_trajectories)
      elif balanced == True and changed_action_space == False:
        print("Testing MIVIA RT-1 on the simulated dataset trained for {} epochs on {} trajectories with balanced sampling with interval set to {}".format(epoch, number_of_trajectories, interval))
        file_result_path = "Results-MIVIA-RT-1-ChangedSampler-Balanced-Epoch-{}-Interval-{}-Traj-{}.txt".format(epoch,interval, number_of_trajectories)
      else :
        print("Testing MIVIA RT-1 on the simulated dataset trained for {} epochs on {} trajectories with final action space with interval set to {}".format(epoch, number_of_trajectories, interval))
        file_result_path = "Results-MIVIA-RT-1-FinalActionSpace-NewDataset-Epoch-{}-Interval-{}-Traj-{}.txt".format(epoch, interval, number_of_trajectories)
      file = open(file_result_path, "w")
      file.write("Testing MIVIA RT-1 on the simulated dataset trained for {} epochs on {} trajectories with interval set to {}".format(epoch, number_of_trajectories, interval))
      file.write("\n")
  
  else :
    
    
    if original:
     
      print("Testing RT-1 on the original dataset trained for {} epochs on {} trajectories".format(0, number_of_trajectories))
      file_result_path = "Results-RT-1-Original-Epoch-{}-Traj-{}.txt".format(0, number_of_trajectories)
      file = open(file_result_path, "w")
      file.write("Testing RT-1 on the original dataset trained for {} epochs on {} trajectories".format(0, number_of_trajectories))
      file.write("\n")
    else : 
      print("Testing MIVIA RT-1 on the simulated dataset trained for {} epochs on {} trajectories".format(0, number_of_trajectories))
      file_result_path = "Results-MIVIA-RT-1-ChangedActionSpace-Epoch-{}-Traj-{}.txt".format(0, number_of_trajectories)
      file = open(file_result_path, "w")
      file.write("Testing MIVIA RT-1 on the simulated dataset trained for {} epochs on {} trajectories".format(0, number_of_trajectories))
      file.write("\n")
       
  model.eval() # Set the model to evaluation mode


  # Action Tokenizer Initialization
  
  action_tokenizer = RT1ActionTokenizer(model._output_tensor_space, 256)

  # Evaluation Metrics Initialization
  accuracy_x = 0
  accuracy_y = 0
  accuracy_z = 0
  accuracy_roll = 0
  accuracy_pitch = 0
  accuracy_yaw = 0
  accuracy_gripper = 0
  number_of_observation = 0

  accuracy_x_interval = 0
  accuracy_y_interval = 0
  accuracy_z_interval = 0
  accuracy_roll_interval = 0
  accuracy_pitch_interval = 0
  accuracy_yaw_interval = 0
  accuracy_gripper_interval = 0

  var_dict = {"Pick green box and place it into the first bin" : 0, "Pick green box and place it into the second bin" : 1, "Pick green box and place it into the third bin" : 2, "Pick green box and place it into the fourth bin" : 3, "Pick yellow box and place it into the first bin" : 4, "Pick yellow box and place it into the second bin" : 5, "Pick yellow box and place it into the third bin" : 6, "Pick yellow box and place it into the fourth bin" : 7, "Pick blue box and place it into the first bin" : 8, "Pick blue box and place it into the second bin" : 9, "Pick blue box and place it into the third bin" : 10, "Pick blue box and place it into the fourth bin" : 11, "Pick red box and place it into the first bin" : 12, "Pick red box and place it into the second bin" : 13, "Pick red box and place it into the third bin" : 14, "Pick red box and place it into the fourth bin" : 15}
  language_embedding_dict = torch.load("/user/mspremulli/Language-Conditioned-Imitation-Learning/data/language_embeddings.pkl", map_location=device)
  with torch.no_grad():



    for traj_path in sampled_trajectories:
        
        
        print("Resetting network state...")
        network_state = batched_space_sampler(model._state_space, batch_size = 1)
        network_state = np_to_tensor(network_state)
        network_state["seq_idx"] = torch.tensor([0])
        network_state["action_tokens"] = network_state["action_tokens"].to(device)
        network_state["context_image_tokens"] = network_state["context_image_tokens"].to(device)
        network_state["seq_idx"] = network_state["seq_idx"].to(device)
        
        
        """
        # Images Check of only the first sampled trajectory
        if traj_path == sampled_trajectories[5]:
          
          # trajectory = torch.load(data_path + traj_path)
          with open(data_path + traj_path, "rb") as f:
            trajectory = pickle.load(f)
          trajectory_data = trajectory["steps"]
          trajectory_length = len(trajectory_data)

          
          for i in range(trajectory_length):
              print("Timestep : {}".format(i))
              image = trajectory_data[i]["observation"]["image"]
              image = image.squeeze()
              pilImg = transforms.ToPILImage()(image)
              pilImg.save("image_timestep_{}.png".format(i))
              action_world_vector = torch.from_numpy(trajectory_data[i]["action"]["world_vector"])
              action_rotation_delta = torch.from_numpy(trajectory_data[i]["action"]["rotation_delta"])
              action_gripper_closedness = torch.from_numpy(trajectory_data[i]["action"]["gripper_closedness_action"])
              print(action_world_vector)
              action = {"world_vector": action_world_vector, "rotation_delta": action_rotation_delta, "gripper_closedness_action": action_gripper_closedness}
              action_tokens = action_tokenizer.tokenize(action)
              print(action_tokens)

              if i == 65:
                print(aojkml)
            
        """
        # trajectory = torch.load(data_path + traj_path)
        with open(data_path + traj_path, "rb") as f:
            trajectory = pickle.load(f)
        trajectory_data = trajectory["steps"]
        trajectory_length = len(trajectory_data)

        accuracy_x_traj = 0
        accuracy_y_traj = 0
        accuracy_z_traj = 0
        accuracy_roll_traj= 0
        accuracy_pitch_traj = 0
        accuracy_yaw_traj = 0
        accuracy_gripper_traj = 0

        accuracy_x_traj_interval = 0
        accuracy_y_traj_interval = 0
        accuracy_z_traj_interval = 0
        accuracy_roll_traj_interval = 0
        accuracy_pitch_traj_interval = 0
        accuracy_yaw_traj_interval = 0
        accuracy_gripper_traj_interval = 0

        natural_language_instruction = trajectory_data[0]["observation"]["natural_language_instruction"]
        file.write("Task : {}".format(natural_language_instruction))
        file.write("\n")
        file.write("Task Length : {}".format(trajectory_length))
        file.write("\n")

          
        print("Task : {}".format(natural_language_instruction))
        print("Length : {}".format(trajectory_length))

        for obs_index in range(trajectory_length):
            
            # Extracting the observation data
            obs_dict = trajectory_data[obs_index]
            image = obs_dict["observation"]["image"]
            image = image.unsqueeze(0)
            natural_language_embedding = obs_dict["observation"]["natural_language_embedding"]
            natural_language_embedding = torch.from_numpy(natural_language_embedding)
            # var_id = var_dict[natural_language_instruction]
            # var_id = var_id + 4
            # print("Current variation : {}".format(var_id))
            # natural_language_embedding = language_embedding_dict[var_id]
            action_x = torch.from_numpy(np.array(obs_dict["action"]["world_vector"][0]))
            action_y = torch.from_numpy(np.array(obs_dict["action"]["world_vector"][1]))
            action_z = torch.from_numpy(np.array(obs_dict["action"]["world_vector"][2]))
            action_roll = torch.from_numpy(np.array(obs_dict["action"]["rotation_delta"][0]))
            action_pitch = torch.from_numpy(np.array(obs_dict["action"]["rotation_delta"][1]))
            action_yaw = torch.from_numpy(np.array(obs_dict["action"]["rotation_delta"][2]))
            action_gripper_closedness = torch.from_numpy(obs_dict["action"]["gripper_closedness_action"])

            # Ground Truth Action
            action = {'x_axis': action_x.to(device), 'y_axis': action_y.to(device), 'z_axis': action_z.to(device), 'roll': action_roll.to(device), 'pitch': action_pitch.to(device), 'yaw': action_yaw.to(device), 'gripper_closedness_action': action_gripper_closedness.to(device)}

            action_tokens = action_tokenizer.tokenize(action)
            print("--- Timestep : {} ---".format(obs_index))
            print("Ground Truth Action Tokens : {}".format(action_tokens))

            # Model Prediction
            observation = {"image": image.to(device), "natural_language_embedding": natural_language_embedding.to(device)}
            predicted_action, network_state = model(observation, network_state)
          

            # Extracting the predicted action tokens
            if obs_index <= 5:
              predicted_action_tokens = network_state["action_tokens"][0][obs_index]
              
            else :
              predicted_action_tokens = network_state["action_tokens"][0][5]
            
            print("Predicted Action Tokens : {}".format(predicted_action_tokens))
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
              accuracy_roll_traj+= 1
              accuracy_roll += 1
            if action_tokens[4] == predicted_action_tokens[4]:
              accuracy_pitch_traj += 1
              accuracy_pitch += 1
            if action_tokens[5] == predicted_action_tokens[5]:
              accuracy_yaw_traj += 1
              accuracy_yaw += 1
            if action_tokens[6] == predicted_action_tokens[6]:
              accuracy_gripper_traj += 1
              accuracy_gripper += 1

            # Interval Evaluation

            if abs(action_tokens[0] - predicted_action_tokens[0]) <= interval:
              accuracy_x_traj_interval += 1
              accuracy_x_interval += 1
            if abs(action_tokens[1] - predicted_action_tokens[1]) <= interval:
              accuracy_y_traj_interval += 1
              accuracy_y_interval += 1
            if abs(action_tokens[2] - predicted_action_tokens[2]) <= interval:
              accuracy_z_traj_interval += 1
              accuracy_z_interval += 1
            if abs(action_tokens[3] - predicted_action_tokens[3]) <= interval:
              accuracy_roll_traj_interval += 1
              accuracy_roll_interval += 1
            if abs(action_tokens[4] - predicted_action_tokens[4]) <= interval:
              accuracy_pitch_traj_interval += 1
              accuracy_pitch_interval += 1
            if abs(action_tokens[5] - predicted_action_tokens[5]) <= interval:
              accuracy_yaw_traj_interval += 1
              accuracy_yaw_interval += 1
            if abs(action_tokens[6] - predicted_action_tokens[6]) <= interval:
              accuracy_gripper_traj_interval += 1
              accuracy_gripper_interval += 1
              
            number_of_observation += 1
            
        # Printing the results

        print("Accuracy X : {}".format(accuracy_x_traj/trajectory_length))
        print("Accuracy Y : {}".format(accuracy_y_traj/trajectory_length))
        print("Accuracy Z : {}".format(accuracy_z_traj/trajectory_length))
        print("Accuracy Rotation X : {}".format(accuracy_roll_traj/trajectory_length))
        print("Accuracy Rotation Y : {}".format(accuracy_pitch_traj/trajectory_length))
        print("Accuracy Rotation Z : {}".format(accuracy_yaw_traj/trajectory_length))
        print("Accuracy Gripper : {}".format(accuracy_gripper_traj/trajectory_length))

        file.write("Accuracy X : {}".format(accuracy_x_traj/trajectory_length))
        file.write("\n")
        file.write("Accuracy Y : {}".format(accuracy_y_traj/trajectory_length))
        file.write("\n")
        file.write("Accuracy Z : {}".format(accuracy_z_traj/trajectory_length))
        file.write("\n")
        file.write("Accuracy Roll : {}".format(accuracy_roll_traj/trajectory_length))
        file.write("\n")
        file.write("Accuracy Pitch : {}".format(accuracy_pitch_traj/trajectory_length))
        file.write("\n")
        file.write("Accuracy Yaw : {}".format(accuracy_yaw_traj/trajectory_length))
        file.write("\n")
        file.write("Accuracy Gripper : {}".format(accuracy_gripper_traj/trajectory_length))
        file.write("\n")

        print("Interval Accuracy X : {}".format(accuracy_x_traj_interval/trajectory_length))
        print("Interval Accuracy Y : {}".format(accuracy_y_traj_interval/trajectory_length))
        print("Interval Accuracy Z : {}".format(accuracy_z_traj_interval/trajectory_length))
        print("Interval Accuracy Roll : {}".format(accuracy_roll_traj_interval/trajectory_length))
        print("Interval Accuracy Pitch : {}".format(accuracy_pitch_traj_interval/trajectory_length))
        print("Interval Accuracy Yaw : {}".format(accuracy_yaw_traj_interval/trajectory_length))
        print("Interval Accuracy Gripper : {}".format(accuracy_gripper_traj/trajectory_length))

        file.write("Interval Accuracy X : {}".format(accuracy_x_traj_interval/trajectory_length))
        file.write("\n")
        file.write("Interval Accuracy Y : {}".format(accuracy_y_traj_interval/trajectory_length))
        file.write("\n")
        file.write("Interval Accuracy Z : {}".format(accuracy_z_traj_interval/trajectory_length))
        file.write("\n")
        file.write("Interval Accuracy Roll : {}".format(accuracy_roll_traj_interval/trajectory_length))
        file.write("\n")
        file.write("Interval Accuracy Pitch : {}".format(accuracy_pitch_traj_interval/trajectory_length))
        file.write("\n")
        file.write("Interval Accuracy Yaw : {}".format(accuracy_yaw_traj_interval/trajectory_length))
        file.write("\n")
        file.write("Interval Accuracy Gripper : {}".format(accuracy_gripper_traj_interval/trajectory_length))
        file.write("\n")
        file.write("\n")
        file.write("\n")

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
    
    file.write("Overall Accuracy X : {}".format(accuracy_x/number_of_observation))
    file.write("\n")
    file.write("Overall Accuracy Y : {}".format(accuracy_y/number_of_observation))
    file.write("\n")
    file.write("Overall Accuracy Z : {}".format(accuracy_z/number_of_observation))
    file.write("\n")
    file.write("Overall Accuracy Roll : {}".format(accuracy_roll/number_of_observation))
    file.write("\n")
    file.write("Overall Accuracy Pitch : {}".format(accuracy_pitch/number_of_observation))
    file.write("\n")
    file.write("Overall Accuracy Yaw : {}".format(accuracy_yaw/number_of_observation))
    file.write("\n")
    file.write("Overall Accuracy Gripper : {}".format(accuracy_gripper/number_of_observation))
    file.write("\n")
    file.write("Overall Interval Accuracy X : {}".format(accuracy_x_interval/number_of_observation))
    file.write("\n")
    file.write("Overall Interval Accuracy Y : {}".format(accuracy_y_interval/number_of_observation))
    file.write("\n")
    file.write("Overall Interval Accuracy Z : {}".format(accuracy_z_interval/number_of_observation))
    file.write("\n")
    file.write("Overall Interval Accuracy Roll : {}".format(accuracy_roll_interval/number_of_observation))
    file.write("\n")
    file.write("Overall Interval Accuracy Pitch : {}".format(accuracy_pitch_interval/number_of_observation))
    file.write("\n")
    file.write("Overall Interval Accuracy Yaw : {}".format(accuracy_yaw_interval/number_of_observation))
    file.write("\n")
    file.write("Overall Interval Accuracy Gripper : {}".format(accuracy_gripper_interval/number_of_observation))
    file.write("\n")
    file.close()


         

          



if __name__ == "__main__":
  evaluations()