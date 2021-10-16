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
    print(X.cpu().numpy())


if __name__ == "__main__":
    dist_sample()
