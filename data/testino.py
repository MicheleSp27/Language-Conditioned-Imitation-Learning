from torch.utils.data import Sampler, RandomSampler
import torch

if __name__ == "__main__":

    lista = [1, 2, 3, 4, 5]
    generator = torch.manual_seed(21)  # Seed for reproducibility

    sampler = RandomSampler(lista, replacement=False)
    sampler_iterable = iter(sampler)

    sampled = next(sampler_iterable)
    print("Sampled element:", sampled)
    sampled = next(sampler_iterable)
    print("Sampled element:", sampled)
    sampled = next(sampler_iterable)
    print("Sampled element:", sampled)

    sampler = RandomSampler(lista, replacement=False)
    sampler_iterable = iter(sampler)

    print("-----------")

    sampled = next(sampler_iterable)
    print("Sampled element:", sampled)
    sampled = next(sampler_iterable)
    print("Sampled element:", sampled)
    sampled = next(sampler_iterable)
    print("Sampled element:", sampled)