

import torch.nn as nn
import torch

## Generator
class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.latend_dim = latent_dim
        self.dim = dim
        
        # self.model = nn.Sequential(
        #     nn.ConvTranspose2d(3, 16, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.25),
        #     nn.Conv2d(16, 32, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.25),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(32, 32, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.25),
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.25),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.25),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.25),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(128, 512, 2),
        #     )
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(512,10),
            )
    
        self.model = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
            )

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
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