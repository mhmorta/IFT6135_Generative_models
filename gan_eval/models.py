

import torch.nn as nn
import torch

## Generator
class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 512, 2),
            )
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(512,10),
            )
    def forward(self, x):
        return self.mlp(self.model(x)[:,:,0, 0])

## Discriminator
class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
            )
    def forward(self, x):
        return self.model(x)