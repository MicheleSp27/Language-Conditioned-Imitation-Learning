#!pip install hydra-core --upgrade
import hydra
import sys
import wandb

sys.path.insert(0,'/user/mspremulli/Language-Conditioned-Imitation-Learning/')

from model.transformer_network import TransformerNetwork
from model.utils import batched_space_sampler,np_to_tensor
from data.dataset3 import CustomDataset3
from data.sampler2 import CustomSampler2
import torch
import numpy as np 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import debugpy


@hydra.main(version_base=None, config_path="", config_name="config2")
def training_procedure(config):
  """
  debugpy.listen(('0.0.0.0', 5678))
  print("Waiting for debugger attach")
  debugpy.wait_for_client()
  """ 
  step = 0 # Training Counter Step
  starting_epoch = 0 # Training Counter Epoch

  if torch.cuda.is_available():
      print(torch.cuda.get_device_name(0))
      device = torch.device("cuda")
      print("Cuda")
  else:
      device = torch.device("cpu")

  pretraining = config.training.load_pretraining
  if pretraining :
    checkpoint_path = config.training.checkpoint_pretraining

  if config.Original:
    print("Initializing Robotic Transformer 1 Model...")
    robotic_transformer_model = hydra.utils.instantiate(config.Transformer_Original)
    if pretraining :
      print("Loading Pretrained Weights...")
      state_dict = torch.load(checkpoint_path)
      robotic_transformer_model.load_state_dict(state_dict["model_state_dict"])
    robotic_transformer_model = robotic_transformer_model.to(device)
    print("Robotic Transformer 1 Model Initialized")
    print("RT-1 Dataset Initialization...")
    dataset_robotic = hydra.utils.instantiate(config.rt1_dataset)
    print("RT-1 Dataset Initialized")
  else:
    print("Initializing Robotic Transformer 1 Model...")
    robotic_transformer_model = hydra.utils.instantiate(config.Transformer_MIVIA)
    if pretraining :
      print("Loading Pretrained Weights...")
      state_dict = torch.load(checkpoint_path)
      robotic_transformer_model.load_state_dict(state_dict["model_state_dict"])
      step = int(checkpoint_path.split("-")[-1].split(".")[0]) * 10
      starting_epoch = state_dict["epoch"]
      print("Starting Step: {}".format(step))
      print("Starting Epoch: {}".format(starting_epoch))
    robotic_transformer_model = robotic_transformer_model.to(device)
    print("Robotic Transformer 1 Model Initialized")
    print("UR5e Dataset Initialization...")
    dataset_robotic = hydra.utils.instantiate(config.mivia_simulated_dataset)
    print("UR5e Dataset Initialized")
  
  # Training Parameters
  batch_size = config.training.batch_size
  epochs = config.training.epochs
  checkpoint = config.training.checkpoint
  checkpoint_path = config.training.checkpoint_save_path
  num_workers = config.training.num_workers


  # Data Sampler Initialization
  number_of_obs_for_trajectory = dataset_robotic._number_of_obs_for_trajectory
  range_observations = dataset_robotic._range_observations
  data_sampler = CustomSampler2(number_of_obs_for_trajectory, range_observations, batch_size)

  # Network State Initialization. This is not used during the training procedure but is a required parameter for the forward pass of the model
  network_state = batched_space_sampler(robotic_transformer_model._state_space, batch_size = batch_size)
  network_state = np_to_tensor(network_state)
  network_state["seq_idx"] = torch.tensor([0])

  """
  number_of_observations = len(dataset_robotic)
  number_of_trajectories = len(available_trajectories)

  print("Number of Trajectories : {}".format(number_of_trajectories))
  print("Number of Observations : {}".format(number_of_observations))
  """

  # Data Loader Initialization
  dataloader_robotic = DataLoader(dataset_robotic, sampler = data_sampler, batch_size = batch_size, num_workers = num_workers)

  # Adam is the optimizer used in the training test from the original work
  optimizer = torch.optim.AdamW(robotic_transformer_model.parameters())

  wandb.init(project = "RT-1 No Pretraining Final Action Space Sampler Changed Simulated Training New Dataset More Placing Different DataLoader", entity = "m-spremulli1-universit-degli-studi-di-salerno")
  
  actions = {}
  for epoch in range(epochs - starting_epoch):

    loss_sum = 0
    batch = 1

    print("Epoch : {}".format(epoch + 1 + starting_epoch))


    for samples in dataloader_robotic:

      images, natural_language_embedding, action_gripper, action_x_axis, action_y_axis, action_z_axis, action_roll, action_pitch, action_yaw = samples
      
      optimizer.zero_grad()

      actions['gripper_closedness_action'] = action_gripper.to(device)
      actions['x_axis'] = action_x_axis.to(device)
      actions['y_axis'] = action_y_axis.to(device)
      actions['z_axis'] = action_z_axis.to(device)
      actions['roll'] = action_roll.to(device)
      actions['pitch'] = action_pitch.to(device)
      actions['yaw'] = action_yaw.to(device)

      """
      # print("X axis shape : {}".format(actions['x_axis'].shape))
      # print("Y axis shape : {}".format(actions['y_axis'].shape))
      # print("Z axis shape : {}".format(actions['z_axis'].shape))
      # print("Roll shape : {}".format(actions['roll'].shape))
      # print("Pitch shape : {}".format(actions['pitch'].shape))
      # print("Yaw shape : {}".format(actions['yaw'].shape))
      """



      observations = {"image":images.to(device), "natural_language_embedding":natural_language_embedding.to(device)}
      
      robotic_transformer_model.set_actions(actions)
      predicted_action, network_state = robotic_transformer_model(observations, network_state)
      
      
      loss = robotic_transformer_model._loss
      
      loss.backward()
      optimizer.step()


      loss_sum = loss_sum + loss.item()

      print("Epoch : {} Batch : {} Loss : {} Loss Epoch : {}".format(epoch + 1 + starting_epoch, batch, loss.item(), loss_sum/batch))


      if batch % 10 == 0:
        wandb.log({"loss" : loss.item(), "batch" : batch, "epoch" : epoch + 1 + starting_epoch})
      
      batch = batch + 1
      step = step + 1

    if (epoch + 1) % checkpoint == 0: 
      print("Saving Checkpoint...")
      torch.save({
            "model_state_dict" : robotic_transformer_model.state_dict(),
            "epoch" : epoch + 1 + starting_epoch,
            "optimizer_state_dict" : optimizer.state_dict(),
            "loss" : loss_sum
        },checkpoint_path + "model_save-{}.pt".format(int(step/10)))
      print("Checkpoint Saved")
    
    wandb.log({"loss_epoch" : loss_sum/batch})
    wandb.save("training_plot.pth")

    

if __name__ == "__main__":
  training_procedure()
  


        




















