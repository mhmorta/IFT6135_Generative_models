from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)


# source: https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x


class VAE(nn.Module):
    def __init__(self, hidden_features):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 512, kernel_size=(5, 5)),
            nn.BatchNorm2d(512),
            nn.ELU(),
            Flatten()
        )

        self.encoder_mean = nn.Linear(in_features=2048, out_features = hidden_features)
        self.encoder_logvar = nn.Sequential(nn.Linear(in_features=2048, out_features = hidden_features))

        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_features, out_features=512),
            nn.ELU(),
            UnFlatten(),
            nn.Conv2d(512, 256, kernel_size=(5, 5), padding=(4, 4)),
            nn.BatchNorm2d(256),
            nn.ELU(),
            Interpolate(scale_factor=2),
            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4, 4)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            Interpolate(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            #Interpolate(scale_factor=2),
            nn.Conv2d(32, 3, kernel_size=(3, 3), padding=(2, 2)),
            nn.BatchNorm2d(3),
            #nn.ELU(),
            #nn.Conv2d(16, 3, kernel_size=(3, 3), padding=(2, 2)),
            #nn.BatchNorm2d(3),



            # nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(2, 2)),
            # nn.BatchNorm2d(64),
            # nn.ELU(),
            # Interpolate(scale_factor=2),
            # nn.Conv2d(64, 3, kernel_size=(3, 3), padding=(2, 2)),
            # nn.BatchNorm2d(3),
            # nn.ELU(),
            # Interpolate(scale_factor=2),
            # nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(2, 2)),
            # nn.BatchNorm2d(16),
            # nn.ELU(),
            # nn.Conv2d(16, 3, kernel_size=(3, 3), padding=(2, 2)),
            nn.Tanh(),
            Flatten(),
            nn.Linear(in_features=3072, out_features=3072),
            nn.Tanh()
        )

        self.flatten = Flatten()

    def encode(self, x):
        h = self.encoder(x)
        return self.encoder_mean(h), self.encoder_logvar(h)

    def reparameterize(self, mu, logvar):
        # convert to the standard deviation
        std = torch.exp(0.5 * logvar)

        # random sampling from normal distribution (0, 1)
        eps = torch.randn_like(std)

        # reparametrize
        return mu + eps * std

    def loss_function(self, x_decoded_mean, x, z_mean, z_logvar):
        x_flatten = self.flatten(x)
        logp_xz = -F.mse_loss(x_flatten, x_decoded_mean, reduction="sum")
        KLD = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).sum()

        # divide by batch size to get average value
        ELBO = (logp_xz - KLD) / x.size(0)

        # optimizer will minimize loss function, thus in order to maximize ELBO we have to negate it, i.e loss = -ELBO
        return -ELBO

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean_z, logvar_z = self.encode(x)
        z = self.reparameterize(mean_z, logvar_z)
        mean_x = self.decode(z)
        return z, mean_x, mean_z, logvar_z

    def generate(self, z):
        return self.decode(z).view(z.size(0), 3, 32, 32)


