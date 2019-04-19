from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 1)


# source: https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode='bilinear')
        return x


class VAE(nn.Module):
    def __init__(self, hidden_features):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=(5, 5)),
            nn.ELU(),
            Flatten()
        )

        self.encoder_mean = nn.Linear(in_features=1024, out_features = hidden_features)
        self.encoder_logvar = nn.Linear(in_features=1024, out_features = hidden_features)

        self.decoder = nn.Sequential(
            nn.Linear(in_features = hidden_features, out_features = 256),
            nn.ELU(),
            UnFlatten(),
            nn.Conv2d(256, 64, kernel_size = (5, 5), padding = (4, 4)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size = (3, 3), padding = (2, 2)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.Conv2d(32, 16, kernel_size = (3, 3), padding = (2, 2)),
            nn.ELU(),
            nn.Conv2d(16, 3, kernel_size = (3, 3), padding = (4, 4)),
        )

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

    def loss_function(self, x, x_decoded_mean, z_mean, z_logvar):
        #x = Flatten()(x)
        #x_decoded_mean = Flatten()(x_decoded_mean)
        log_likelihood = - F.mse_loss(x, x_decoded_mean, reduction='sum') # / (2 * 0.01 ** 2) + math.log(0.01)
        KLD = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        ELBO = log_likelihood - KLD

        # optimizer will minimize loss function, thus in order to maximize ELBO we have to negate it, i.e loss = -ELBO
        return -ELBO / x.size(0)

    def decode(self, z):
        return torch.tanh(self.decoder(z))

    def forward(self, x):
        mean_z, logvar_z = self.encode(x)
        z = self.reparameterize(mean_z, logvar_z)
        mean_x = self.decode(z)
        return z, mean_x, mean_z, logvar_z

    def generate(self, z):
        return self.decode(z)


