import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x


class Generator(nn.Module):
    def __init__(self, channels, latent_dim, cuda):
        super(Generator, self).__init__()
        self.cuda = cuda

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
#             nn.ReLU()
            )

        self.convTranspose = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
#             nn.Dropout2d(0.25),
            nn.ELU(),

            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
#             nn.Dropout2d(0.25),
            nn.ELU(),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
#             nn.Dropout2d(0.25),
            nn.ELU(),

            nn.ConvTranspose2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            # nn.Dropout2d(0.25),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
#             nn.Dropout2d(0.5),
            nn.ELU(),

            nn.Conv2d(8, channels, 3, 1, 1),
            nn.Tanh()
            )
        self.init_weights()

    def init_weights(self):
        for m in self.convTranspose:
            if isinstance(m,nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if type(self.mlp) == nn.Linear:
            nn.init.xavier_uniform_(self.mlp.weight)
            self.mlp.bias.data.fill_(0.01)

    def forward(self, input):
        x = self.mlp(input)
        x = x.view(-1, 128, 4, 4)
        return self.convTranspose(x)




class Discriminator(nn.Module):
    def __init__(self, channels, latent_dim, cuda):
        super(Discriminator, self).__init__()
        self.cuda = cuda

        self.conv = nn.Sequential( 
            nn.Conv2d(channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
#             nn.Dropout2d(0.25),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
#             nn.Dropout2d(0.25),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
#             nn.Dropout2d(0.5),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
#             nn.Dropout2d(0.25),
            nn.ReLU(),

            # nn.Conv2d(32, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.Dropout2d(0.25),
            # nn.LeakyReLU(0.2),

            # nn.Conv2d(64, 128, 3, 2, 1),
            # nn.BatchNorm2d(128),
            # nn.Dropout2d(0.5),
            # nn.LeakyReLU(0.2),            
            )
        self.mlp = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512,1)
#             nn.Sigmoid()
            )

        self.init_weights()
    def init_weights(self):
        for m in self.conv:
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if type(self.mlp) == nn.Linear:
            nn.init.xavier_uniform_(self.mlp.weight)
            self.mlp.bias.data.fill_(0.01)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * 4 * 4)
        return self.mlp(x)

