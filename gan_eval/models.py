

import torch.nn as nn
import torch

## Generator
class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self,latent_dim):
        super(Generator, self).__init__()
        self.img_size = 32
        self.latend_dim = latent_dim
        self.dim = 16
        self.feature_sizes = (self.img_size/16, self.img_size/16)

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            )
       
    
        self.model = nn.Sequential(
            nn.ConvTranspose2d(8 * self.dim, 4 * self.dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * self.dim),
            nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * self.dim),
            nn.ConvTranspose2d(2 * self.dim, self.dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.dim),
            nn.ConvTranspose2d(self.dim, self.img_size, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            )


    def forward(self, input):
        print (input.size())
        x = self.mlp(input)
        x = x.view(-1, 8 * self.dim, 4, 4)
        print (x.shape[0], x.shape[1])
        return self.model(x)
    # def sample_latent(self, num_samples):
    #     return torch.randn((num_samples, self.latent_dim))

## Discriminator
class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, channel, latent_dim):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.channel = channel
        self.model = nn.Sequential(
            nn.Conv2d(self.channel, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
            )
        self.mlp = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1)
            )
    def forward(self, input):
        x = self.model(input)
        x = x.view(-1, 128 * 4 * 4)
        return self.mlp(x)



 # 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]