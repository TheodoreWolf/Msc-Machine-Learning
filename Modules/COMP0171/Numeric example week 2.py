import torch.distributions as dist
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda')


def bayes_example():
    """
    Slide numeric exercise from mackay p25
    """
    pa = 0.01
    pb = 0.95
    pab = pa * pb / (pa * pb + (1 - pa) * (1 - pb))
    return pab


def dist_sample():
    X = dist.Beta(5, 4).sample().to(device)
    print(X.device)
    print(X.cpu().numpy())


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        print(torch.cuda.device_count())
        print(torch.cuda.is_available())
        
