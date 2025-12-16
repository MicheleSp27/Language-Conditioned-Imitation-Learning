# Script used for RT-1 pre-training on the original dataset.
import sys
sys.path.insert(0,'/user/mspremulli/Language-Conditioned-Imitation-Learning/')

from model.transformer_network import TransformerNetwork # RT-1 Model
from model.utils import batched_space_sampler,np_to_tensor # Utilities functions
from data.dataset import CustomDataset # Pytorch Custom Dataset Class
from data.sampler import CustomSampler # Python Custom Sampler Class
import torch 
import numpy as np 
from torch.utils.data import DataLoader
import wandb
import hydra


@hydra.main(version_base=None, config_path="", config_name="config")
def training_procedure(config):

  # Setting GPU for the training procedure.
  if torch.cuda.is_available():
      print("Using GPU {} for training.".format(torch.cuda.get_device_name(0)))
      device = torch.device("cuda")
  else:
      print("Using CPU for training.")
      device = torch.device("cpu")

  # Model and Dataset Initialization
  print("Initializing RT-1 Model with the original action space...")
  robotic_transformer_model = hydra.utils.instantiate(config.Transformer_Original) # Instantiation of the RT-1 model from the config file.
  robotic_transformer_model = robotic_transformer_model.to(device) # Moving the model to the selected device.
  print("RT-1 Model Initialized") 
  print("RT-1 real dataset initialization...")
  dataset_robotic = hydra.utils.instantiate(config.rt1_dataset) # Instantiation of the RT-1 dataset from the config file.
  print("RT-1 Dataset Initialized")

  # Training Parameters
  batch_size = config.training.batch_size # Size of the training batch.
  epochs = config.training.epochs # Number of training epochs.
  checkpoint = config.training.checkpoint # Every checkpoint epochs a model checkpoint is saved.
  checkpoint_path = config.training.checkpoint_save_path # Path were the checkpoints are saved.
  num_workers = config.training.num_workers # Number of workers for the data loader.
  weight_bias = config.training.weight_bias # Boolean value indicating whether to use W&B for experiment tracking.
  if weight_bias:
    project_name = config.training.project_name # W&B Project name.
    entity = config.training.entity # W&B Entity name.

  # Data Sampler Initialization
  trajectory_obs_id = dataset_robotic._trajectory_obs_id # Dictionary with key the trajectory index and value a list with the observation ids of the trajectory.
  number_of_obs_for_trajectory = dataset_robotic._number_of_obs_for_trajectory # Dictionary with key the trajectory index and value the number of observations in the trajectory.
  data_sampler = CustomSampler(trajectory_obs_id, number_of_obs_for_trajectory, batch_size) # Instantiation of the custom sampler.

  # Network State Initialization. This is not used during the training procedure but is a required parameter for the forward pass of the model.
  network_state = batched_space_sampler(robotic_transformer_model._state_space, batch_size = batch_size)
  network_state = np_to_tensor(network_state)
  network_state["seq_idx"] = torch.tensor([0])

  # Data Loader Initialization.
  dataloader_robotic = DataLoader(dataset_robotic, sampler = data_sampler, batch_size = batch_size, num_workers = num_workers)

  # Adam is the optimizer used in the training test from the original work.
  optimizer = torch.optim.AdamW(robotic_transformer_model.parameters())

  if weight_bias:
    wandb.init(project = project_name, entity = entity)
  
  step = 0 # Training Counter Step
  actions = {} # Dictionary used to store ground truth actions for the forward pass of the model.

  # Training Loop.
  for epoch in range(epochs):

    loss_sum = 0
    batch = 1
    
    print("Epoch : {}".format(epoch + 1))

    for samples in dataloader_robotic:

      images, natural_language_embedding, action_gripper, action_rotation_delta, action_world_vector = samples # Retrieving data from the dataloader.
    
      optimizer.zero_grad() # Zeroing the gradients of the optimizer.

      # Setting ground truth actions.
      actions["gripper_closedness_action"] = action_gripper.to(device) 
      actions["world_vector"] = action_world_vector.to(device)
      actions["rotation_delta"] = action_rotation_delta.to(device)

      # Moving input observations to the selected device.
      observations = {"image":images.to(device), "natural_language_embedding":natural_language_embedding.to(device)}
      
      # Setting ground truth actions.
      robotic_transformer_model.set_actions(actions)
      # Forward pass of the model.
      predicted_action, network_state = robotic_transformer_model(observations, network_state)
      # Retrieving the loss value.
      loss = robotic_transformer_model._loss
      # Loss backward pass and optimizer step.
      loss.backward()
      optimizer.step()

      loss_sum = loss_sum + loss.item()
      print("Epoch : {} Batch : {} Loss : {} Loss Epoch : {}".format(epoch + 1, batch, loss.item(), loss_sum/batch))

      # Logging training information to W&B every 10 batches.
      if batch % 10 == 0 and weight_bias:
        wandb.log({"loss" : loss.item(), "batch" : batch, "epoch" : epoch + 1})
      
      batch = batch + 1
      step = step + 1
    # Saving checkpoints.
    if (epoch + 1) % checkpoint == 0: 
      print("Saving Checkpoint...")
      torch.save({
            "model_state_dict" : robotic_transformer_model.state_dict(),
            "epoch" : epoch + 1,
            "optimizer_state_dict" : optimizer.state_dict(),
            "loss" : loss_sum
        },checkpoint_path + "model_save-{}.pt".format(int(step/10)))
      print("Checkpoint Saved")
    
    if weight_bias:
      wandb.log({"loss_epoch" : loss_sum/batch})
      wandb.save("training_plot.pth")

if __name__ == "__main__":
  training_procedure()
  


        




















