# Script used for RT-1 finetuning on MIVIA simulated data
import sys
sys.path.insert(0,'/user/mspremulli/Language-Conditioned-Imitation-Learning/')

from model.transformer_network import TransformerNetwork # RT-1 Model
from model.utils import batched_space_sampler,np_to_tensor # Utilities functions
from data.dataset2 import CustomDataset2 # Pytorch Custom Dataset Class
from data.sampler2 import CustomSampler2 # Python Custom Sampler Class
import torch
from torch.utils.data import DataLoader
import wandb
import hydra

@hydra.main(version_base=None, config_path="", config_name="config2")
def training_procedure(config):

  # Setting GPU for the training procedure.
  if torch.cuda.is_available():
      print("Using GPU {} for training.".format(torch.cuda.get_device_name(0)))
      device = torch.device("cuda")
  else:
      print("Using CPU for training.")
      device = torch.device("cpu")

  # Training parameters.
  batch_size = config.training.batch_size # Size of the training batch.
  epochs = config.training.epochs # Number of training epochs.
  checkpoint = config.training.checkpoint # Every checkpoint epochs a model checkpoint is saved.
  checkpoint_save = config.training.checkpoint_save_path # Path were the checkpoints are saved.
  num_workers = config.training.num_workers # Number of workers for the data loader.
  pretraining = config.training.pretraining # Boolean value indicating whether to load pretrained weights or not. If false, model is trained from scratch. 
  if pretraining:
    checkpoint_path = config.training.checkpoint_pretraining # Path to the pretrained weights.
  weight_bias = config.training.weight_bias # Boolean value indicating whether to use W&B for experiment tracking.
  if weight_bias:
    project_name = config.training.project_name # W&B Project name.
    entity = config.training.entity # W&B Entity name.

  # Model and Checkpoint Initialization
  print("Initializing RT-1 with custom action space...")
  robotic_transformer_model = hydra.utils.instantiate(config.Transformer_MIVIA)
  print("RT-1 Model Initialized")
  if pretraining :
    print("Loading Pretrained Weights...")
    state_dict = torch.load(checkpoint_path)
    robotic_transformer_model.load_state_dict(state_dict["model_state_dict"])
    robotic_transformer_model = robotic_transformer_model.to(device) # Moving the model to the selected device.
    print("Pretrained for {} epochs :".format(state_dict["epoch"]))
    print("Pretrained Weights Loaded")
  else :
    robotic_transformer_model = robotic_transformer_model.to(device) # Moving the model to the selected device.
    print("Training from scratch")
  
  # Dataset Initialization.
  print("UR5e Dataset Initialization...")
  dataset_robotic = hydra.utils.instantiate(config.mivia_simulated_dataset) # Instantiation of the UR5e dataset from the config file.
  excluded_tasks = config.mivia_simulated_dataset.excluded_tasks # List of excluded tasks in the UR5e dataset.
  batch_size = batch_size - len(excluded_tasks) * 2 # Adjusting batch size based on excluded tasks.
  print("UR5e Dataset Initialized")
  
  # Data Sampler Initialization
  number_of_obs_for_trajectory = dataset_robotic._number_of_obs_for_trajectory
  range_observations = dataset_robotic._range_observations
  data_sampler = CustomSampler2(number_of_obs_for_trajectory, range_observations, batch_size)

  # Network State Initialization. This is not used during the training procedure but is a required parameter for the forward pass of the model
  network_state = batched_space_sampler(robotic_transformer_model._state_space, batch_size = batch_size)
  network_state = np_to_tensor(network_state)
  network_state["seq_idx"] = torch.tensor([0])

  # Data Loader Initialization
  dataloader_robotic = DataLoader(dataset_robotic, sampler = data_sampler, batch_size = batch_size, num_workers = num_workers)

  # Adam is the optimizer used in the training test from the original work
  optimizer = torch.optim.AdamW(robotic_transformer_model.parameters())

  if weight_bias:
    wandb.init(project = project_name, entity = entity)
  
  actions = {} # Dictionary used to store ground truth actions for the forward pass of the model.
  step = 0 # Training Counter Step

  for epoch in range(epochs):

    loss_sum = 0
    batch = 1

    print("Epoch : {}".format(epoch + 1))

    for samples in dataloader_robotic:

      images, natural_language_embedding, action_gripper, action_x_axis, action_y_axis, action_z_axis, action_roll, action_pitch, action_yaw = samples # Retrieving data from the dataloader.
      
      optimizer.zero_grad() # Zeroing the gradients of the optimizer.

      # Setting ground truth actions.
      actions['gripper_closedness_action'] = action_gripper.to(device)
      actions['x_axis'] = action_x_axis.to(device)
      actions['y_axis'] = action_y_axis.to(device)
      actions['z_axis'] = action_z_axis.to(device)
      actions['roll'] = action_roll.to(device)
      actions['pitch'] = action_pitch.to(device)
      actions['yaw'] = action_yaw.to(device)

      # Moving input observations to the selected device.
      observations = {"image":images.to(device), "natural_language_embedding":natural_language_embedding.to(device)}
      robotic_transformer_model.set_actions(actions) # Setting ground truth actions
      # Forward pass of the model.
      predicted_action, network_state = robotic_transformer_model(observations, network_state)
      # Retrieving the loss value.
      loss = robotic_transformer_model._loss
      
      # Loss Backward Pass and Optimization Step.
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
        },checkpoint_save + "model_save-{}.pt".format(int(step/10)))
      print("Checkpoint Saved")
    
    if weight_bias:
      wandb.log({"loss_epoch" : loss_sum/batch})
      wandb.save("training_plot.pth")

if __name__ == "__main__":
  training_procedure()
  


        




















