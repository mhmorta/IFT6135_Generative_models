from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size = (3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(64, 256, kernel_size = (5, 5)),
            nn.ELU(),
            nn.Linear(in_features = 256, out_features = 100)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features = 100, out_features = 256),
            nn.ELU(),
            nn.Conv2d(256, 64, kernel_size = (5, 5), padding = (4, 4)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.Conv2d(64, 32, kernel_size = (3, 3), padding = (2, 2)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.Conv2d(32, 16, kernel_size = (3, 3), padding = (2, 2)),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size = (3, 3), padding = (2, 2))
        )

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # TODO confirm if we do need this?
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #return BCE + KLD

    return BCE
