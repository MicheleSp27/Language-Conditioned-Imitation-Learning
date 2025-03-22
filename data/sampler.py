from torch.utils.data import Sampler, RandomSampler
import numpy as np
import torch

class CustomSampler(Sampler):
    
  def __init__(self, available_trajectories, trajectory_obs_id, number_of_obs_for_trajectory, batch_size):
    
      self._available_trajectories = available_trajectories # List with indexs of the available trajectories
      self._trajectory_obs_id = trajectory_obs_id # Dictionary with key the trajectory index and value a list with the observation ids of the trajectory
      self._number_of_obs_for_trajectory = number_of_obs_for_trajectory # Dictionary with key the trajectory index and value the number of observations in the trajectory
      self._batch_size = batch_size # Batch size
      self._generator = torch.manual_seed(42) # Generator for torch.randint

  def __len__(self):
    pass

  def __iter__(self):

    trajectory_dict = {} # Dictionary containing available trajectories
    trajectory_list = [] # List containing available trajectories
    terminated_trajectory = [] # List containing terminated trajectories
    terminated_trajectory_keys = [] # List containing the keys of the terminated trajectories to be removed from the dictionary trajectory_dict
    number_of_observations_for_trajectory = {} # Copy of the dictionary self._number_of_obs_for_trajectory

    # Initialization of the dictionaries of the available trajectories and the number of observations for each trajectory
    for trajectory_index in self._available_trajectories:
      trajectory_dict[trajectory_index] = True

    for trajectory_index in self._number_of_obs_for_trajectory.keys():
      number_of_observations_for_trajectory[trajectory_index] = self._number_of_obs_for_trajectory[trajectory_index]

    trajectory_list = list(trajectory_dict.keys())

    # Trajectory Sampler Initialization. Used to sample a trajectory from the available ones.
    trajectory_sampler = RandomSampler(trajectory_list, replacement = False)
    trajectory_sampler_iterable = iter(trajectory_sampler)

    # Dict of Sampler over the observations of each trajectory
    observation_iterable = {}

    # List of Sampler initialization
    for trajectory_index in self._available_trajectories:
      observation_list = self._trajectory_obs_id[trajectory_index]
      sampler = RandomSampler(observation_list, replacement = False)
      observation_iterable[trajectory_index] = iter(sampler)

    iteration_sampler = 0 # Number of sampling

    # While there are available trajectories
    while len(trajectory_list) > 0:

      flag = False
      counter_missing = 0

      # The sampler over the available trajectories might reset while the batch contains already sampled observations.
      # This can cause the sampling from trajectories that are already sampled and it breaks the condition of unique trajectories in the batch.
      # To solve this, we force the reset of the sampler.
      # If the number of total sampling less the number of sampling already done is less than the batch size
      # then the problem appears and we need to reset the sampler.

      if len(trajectory_list) - iteration_sampler < self._batch_size:

        # If there are terminated trajectories, we remove from the dictionary of the available trajectories
        if len(terminated_trajectory_keys) > 0:

            for index in terminated_trajectory_keys:
              trajectory_dict.pop(index)

            terminated_trajectory_keys = []

        trajectory_list = list(trajectory_dict.keys())
        if (len(trajectory_list) == 0):
            break
        sampler = RandomSampler(trajectory_list, replacement = False)
        trajectory_sampler_iterable = iter(sampler)
        iteration_sampler = 0

       
      # Batch filling loop

      for batch_index in range(self._batch_size):

        # If the number of available trajectories is smaller than the batch size, we need to sample from the terminated trajectories.
        # We sample missing_samples observations from the terminated trajectories and we add them to the batch.
        if flag == False and len(trajectory_list) < self._batch_size:
          flag = True
          missing_samples = self._batch_size - len(trajectory_list)
          sampler_terminated_trajectory = RandomSampler(terminated_trajectory, replacement = False)
          iterator_terminated_trajectory = iter(sampler_terminated_trajectory)

          for _ in range(missing_samples):

            sampled_index = next(iterator_terminated_trajectory)
            sampled_terminated_trajectory = terminated_trajectory[sampled_index]

            observation_list = self._trajectory_obs_id[sampled_terminated_trajectory]
            sampled_index_missing_observation = torch.randint(low = 0, high = len(observation_list), size = (1,), generator = self._generator)[0]
            sampled_missing_observation = observation_list[sampled_index_missing_observation]

            yield sampled_missing_observation 
            
        # Jump missing_samples iteration because there are already missing_samples observations in the batch
        if flag == True:

          if counter_missing < missing_samples:

            counter_missing = counter_missing + 1
            continue

        # Sampling of the trajectory
        try:
          trajectory_index = next(trajectory_sampler_iterable)
        except StopIteration:
          # If there are terminated trajectories, we remove from the dictionary of the available trajectories
          if len(terminated_trajectory_keys) > 0:

            for index in terminated_trajectory_keys:
              print(trajectory_dict.pop(index))

            terminated_trajectory_keys = []

          iteration_sampler = 0
          trajectory_list = list(trajectory_dict.keys())
      
          if (len(trajectory_list) == 0):
            break
          sampler = RandomSampler(trajectory_list, replacement = False)
          trajectory_sampler_iterable = iter(sampler)
          trajectory_index = next(trajectory_sampler_iterable)

        sampled_trajectory = trajectory_list[trajectory_index]

        # Sampling of the observation
        sampled_index_observation = next(observation_iterable[sampled_trajectory])
        sampled_observation = self._trajectory_obs_id[sampled_trajectory][sampled_index_observation]

        # Updating the counter of the number of observations available for the trajectory
        counter = number_of_observations_for_trajectory[sampled_trajectory]
        counter = counter - 1
        number_of_observations_for_trajectory[sampled_trajectory] = counter

        # If the number of observations available for the trajectory is 0, we add the trajectory to the terminated trajectories
        if (counter == 0):

          terminated_trajectory_keys.append(sampled_trajectory)
          terminated_trajectory.append(sampled_trajectory)
        

        iteration_sampler = iteration_sampler + 1
        yield sampled_observation

    """
    not_finished_trajectory = []

    for key in number_of_observations_for_trajectory:
      if number_of_observations_for_trajectory[key] != 0:
        not_finished_trajectory.append(key)
    
    print("Traiettorie non finite : ")
    print(not_finished_trajectory)
    """
      
