#!pip install hydra-core --upgrade
import hydra
import sys
import wandb

sys.path.insert(0,'/user/mspremulli/Language-Conditioned-Imitation-Learning/')

from model.transformer_network import TransformerNetwork
from model.utils import batched_space_sampler,np_to_tensor
from data.dataset import CustomDataset
from data.sampler import CustomSampler
import torch
import numpy as np 
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="", config_name="config")
def training_procedure(config):

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
      state_dict = torch.load(checkpoint_path)
      robotic_transformer_model.load_state_dict(state_dict["model_state_dict"])
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
  available_trajectories = dataset_robotic._available_trajectories
  trajectory_obs_id = dataset_robotic._trajectory_obs_id
  number_of_obs_for_trajectory = dataset_robotic._number_of_obs_for_trajectory
  data_sampler = CustomSampler(available_trajectories, trajectory_obs_id, number_of_obs_for_trajectory, batch_size)

  # Network State Initialization. This is not used during the training procedure but is a required parameter for the forward pass of the model
  network_state = batched_space_sampler(robotic_transformer_model._state_space, batch_size = batch_size)
  network_state = np_to_tensor(network_state)
  network_state["seq_idx"] = torch.tensor([0])

  number_of_observations = len(dataset_robotic)
  number_of_trajectories = len(available_trajectories)

  print("Number of Trajectories : {}".format(number_of_trajectories))
  print("Number of Observations : {}".format(number_of_observations))

  # Data Loader Initialization
  dataloader_robotic = DataLoader(dataset_robotic, sampler = data_sampler, batch_size = batch_size, num_workers = num_workers)

  # Adam is the optimizer used in the training test from the original work
  optimizer = torch.optim.AdamW(robotic_transformer_model.parameters())

  wandb.init(project = "RT-1 No Pretraining Simulated Training", entity = "m-spremulli1-universit-degli-studi-di-salerno")

  actions = {}
  for epoch in range(epochs):

    loss_sum = 0
    batch = 1

    print("Epoch : {}".format(epoch + 1))

    for samples in dataloader_robotic:

      images, natural_language_embedding, action_gripper, action_rotation_delta, action_world_vector = samples
      
      
      optimizer.zero_grad()

      actions["gripper_closedness_action"] = action_gripper.to(device)
      actions["world_vector"] = action_world_vector.to(device)
      actions["rotation_delta"] = action_rotation_delta.to(device)

      
      observations = {"image":images.to(device), "natural_language_embedding":natural_language_embedding.to(device)}

      robotic_transformer_model.set_actions(actions)
      predicted_action, network_state = robotic_transformer_model(observations, network_state)

      loss = robotic_transformer_model._loss
      
      loss.backward()
      optimizer.step()


      loss_sum = loss_sum + loss.item()

      print("Epoch : {} Batch : {} Loss : {} Loss Epoch : {}".format(epoch + 1, batch, loss.item(), loss_sum))


      if batch % 10 == 0:
        wandb.log({"loss" : loss.item(), "batch" : batch, "epoch" : epoch + 1})
      
      batch = batch + 1

    if (epoch + 1) % checkpoint == 0: 
      print("Saving Checkpoint...")
      wandb.log({"loss_epoch" : loss_sum})
      torch.save({
            "model_state_dict" : robotic_transformer_model.state_dict(),
            "epoch" : epoch + 1,
            "optimizer_state_dict" : optimizer.state_dict(),
            "loss" : loss_sum
        },checkpoint_path + "rt-1_pretraining_simulated_checkpoint_epoch_{}_loss_{}_batch_{}".format(epoch + 1, loss_sum, batch_size))
      print("Checkpoint Saved")
    
    wandb.save("training_plot.pth")

if __name__ == "__main__":
  training_procedure()


        




















