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
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 512, kernel_size=(5, 5)),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            Flatten()
        )

        self.encoder_mean = nn.Linear(in_features=2048, out_features = hidden_features)

        # apply softplus activation function to ensure that variance is positive
        self.encoder_var = nn.Sequential(nn.Linear(in_features=2048, out_features = hidden_features), nn.Softplus())

        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_features, out_features=512),
            nn.Tanh(),
            UnFlatten(),
            nn.Conv2d(512, 256, kernel_size=(5, 5), padding=(4, 4)),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            Interpolate(scale_factor=2),
            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4, 4)),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            Interpolate(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 3, kernel_size=(3, 3), padding=(2, 2)),
            nn.BatchNorm2d(3),
            nn.Tanh(),
            Flatten(),
            nn.Linear(in_features=3072, out_features=3072),

            # clip mean (-1, 1)
            nn.Tanh()
        )

        self.flatten = Flatten()

    def encode(self, x):
        h = self.encoder(x)
        return self.encoder_mean(h), self.encoder_var(h)

    def reparameterize(self, mu, var):

        # sigma = sqrt(var)
        std = var.pow(0.5)

        # random sampling from normal distribution (0, 1)
        eps = torch.randn_like(std)

        # reparametrize
        return mu + eps * std

    def loss_function(self, x_decoded_mean, x, z, z_mean, z_var):

        x_flatten = self.flatten(x)
        logp_xz = -F.mse_loss(x_flatten, x_decoded_mean, reduction="sum")

        # KL divergence between prior of z (0, 1) and approximate posterior of z (z_mean, z_var)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        loss_kl = -0.5 * (1 + z_var.log() - z_mean.pow(2) - z_var).sum()

        # divide by batch size to get average value
        ELBO = (logp_xz - loss_kl) / x.size(0)

        # optimizer will minimize loss function, thus in order to maximize ELBO we have to negate it, i.e loss = -ELBO
        return -ELBO

    def decode(self, z):
        #return self.decoder(z)
        z #torch.Size([64, 100])
        z = nn.Linear(in_features=100, out_features=512).forward(z) #torch.Size([64, 512])
        z = nn.Tanh().forward(z)
        z = UnFlatten().forward(z) #torch.Size([64, 512, 1, 1])
        z = nn.Conv2d(512, 256, kernel_size=(5, 5), padding=(4, 4)).forward(z) #torch.Size([64, 256, 5, 5])
        z = nn.BatchNorm2d(256).forward(z)
        z = nn.Tanh().forward(z)
        z = Interpolate(scale_factor=2).forward(z) #torch.Size([64, 256, 10, 10])
        z = nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4, 4)).forward(z) # torch.Size([64, 64, 14, 14])
        z = nn.BatchNorm2d(64).forward(z)
        z = nn.Tanh().forward(z)
        z = Interpolate(scale_factor=2).forward(z) #torch.Size([64, 64, 28, 28])
        z = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)).forward(z) # torch.Size([64, 32, 30, 30])
        z = nn.BatchNorm2d(32).forward(z)
        z = nn.Tanh().forward(z)
        z = nn.Conv2d(32, 3, kernel_size=(3, 3), padding=(2, 2)).forward(z) #torch.Size([64, 3, 32, 32])
        z = nn.BatchNorm2d(3).forward(z)
        z = nn.Tanh().forward(z)
        z = Flatten().forward(z) # torch.Size([64, 3072])
        z = nn.Linear(in_features=3072, out_features=3072).forward(z) # torch.Size([64, 3072])
        return z


    def forward(self, x):
        mean_z, var_z = self.encode(x)
        z = self.reparameterize(mean_z, var_z)
        mean_x = self.decode(z)
        return z, mean_x, mean_z, var_z

    def generate(self, z):
        return self.decode(z).view(z.size(0), 3, 32, 32)


