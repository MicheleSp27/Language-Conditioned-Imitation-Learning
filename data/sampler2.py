# Sampler Custom Class used for simulated MIVIA Dataset
from torch.utils.data import Sampler, RandomSampler
import numpy as np
import torch

class CustomSampler2(Sampler):

    def __init__(self, number_of_obs_for_trajectory, range_observations, batch_size):

        self._number_of_obs_for_trajectory = number_of_obs_for_trajectory # Dictionary with key the trajectory index and value the number of observations in the trajectory.
        self._range_observations = range_observations # Dictionary with key the trajectory index and value the list of observation ids of the trajectory.
        self._batch_size = batch_size # Batch size.
        self._generator = torch.manual_seed(39) # Generator for torch.randint. It ensures reproducibility of experiments.

    def __len__(self):
        pass # The number of batch is unknown.

    def __iter__(self):

        terminated_trajectories = [] # List containing the index of the terminated trajectories to be removed from the dictionary 
        terminated_trajectories_to_sample = [] # List containing the terminated trajectories to sample observations from them when there are not enough available trajectories to fill the batch.
        number_of_obs_for_trajectory = {} # Copy of self._number_of_obs_for_trajectory

        # Create a copy of the number of observations for each trajectory.
        for i in self._number_of_obs_for_trajectory.keys():
            number_of_obs_for_trajectory[i] = self._number_of_obs_for_trajectory[i]
        trajectory_list = list(number_of_obs_for_trajectory.keys()) # List of the available trajectories
        
        # Initialization of the trajectory sampler
        trajectory_sampler = RandomSampler(trajectory_list, replacement = False)
        trajectory_sampler_iterable = iter(trajectory_sampler)

        # Initialization of the observation sampler
        sampler_dict = {} # Dictionary with the sampler for each trajectory
        for i in trajectory_list:
            sampler = RandomSampler(self._range_observations[i], replacement = False)
            sampler_dict[i] = iter(sampler)

        while len(trajectory_list) > 0: # While there is a trajectory available
            
            flag = False
            iteration_jumped = 0

            for i in range(self._batch_size):

                # If the number of available trajectories can't fill the batch, then observations are reteieved from the terminated trajectories
                if len(trajectory_list) < self._batch_size and flag == False and i == len(trajectory_list):
            
                    flag = True

                    missing_observations = self._batch_size - len(trajectory_list) # Number of missing observations.
                    terminated_trajectories_sampler = RandomSampler(terminated_trajectories_to_sample, replacement = False) # Sampler over the terminated trajectories.
                    terminated_trajectories_sampler_iterable = iter(terminated_trajectories_sampler)

                    for _ in range(missing_observations):
                        terminated_trajectories_sampled_index = next(terminated_trajectories_sampler_iterable)
                        terminated_trajectories_sampled = terminated_trajectories_to_sample[terminated_trajectories_sampled_index]

                        observations_list = self._range_observations[terminated_trajectories_sampled]
                        observation_sampler = RandomSampler(observations_list, replacement = False)
                        observation_sampler_iterable = iter(observation_sampler)
                        observation_index = next(observation_sampler_iterable)
                        observation_sampled = self._range_observations[terminated_trajectories_sampled][observation_index]

                        yield observation_sampled

                # The batch is already filled with missing_observations samples. So jump missing_observations iterations
                if flag == True:
                    iteration_to_jump = self._batch_size - len(trajectory_list)
                    if iteration_jumped < iteration_to_jump:
                        iteration_jumped += 1
                        continue
                
                try:
                    trajectory_index = next(trajectory_sampler_iterable) # Sample a trajectory.
                except StopIteration:
                    # Reset the sampler and delete terminated trajectories from available ones.
                    if len(terminated_trajectories) > 0:

                        for index in terminated_trajectories:
                            number_of_obs_for_trajectory.pop(index)
                        
                        terminated_trajectories = []

                    trajectory_list = list(number_of_obs_for_trajectory.keys())
                    if len(trajectory_list) == 0: # If there are no available trajectories left, the sampler stops.
                        break
                    trajectory_sampler = RandomSampler(trajectory_list, replacement = False)
                    trajectory_sampler_iterable = iter(trajectory_sampler)

                    trajectory_index = next(trajectory_sampler_iterable) # Sample a trajectory.

                # Get the sampled trajectory.
                trajectory_sampled = trajectory_list[trajectory_index]

                # Sampling of the observation.
                trajectory_sampler_observations = sampler_dict[trajectory_sampled]
                observation_index = next(trajectory_sampler_observations)
                observation_sampled = self._range_observations[trajectory_sampled][observation_index]

                # Updating the counter of the number of observations available for the trajectory.
                number_of_observations = number_of_obs_for_trajectory[trajectory_sampled]
                number_of_observations = number_of_observations - 1
                number_of_obs_for_trajectory[trajectory_sampled] = number_of_observations

                # If the number of observations available for the trajectory is 0, we add the trajectory to the terminated trajectories.
                if number_of_observations == 0:
                    terminated_trajectories.append(trajectory_sampled)
                    terminated_trajectories_to_sample.append(trajectory_sampled)

                yield observation_sampled

        # Uncomment to check whether all trajectories are seen.
        """
        not_finished_trajectory = []
        for key in number_of_obs_for_trajectory:
            if number_of_obs_for_trajectory[key] != 0:
                not_finished_trajectory.append(key)
        print("Not finished trajectories : {}".format(not_finished_trajectory))

        print("Original Dict : {}".format(self._number_of_obs_for_trajectory))
        """
