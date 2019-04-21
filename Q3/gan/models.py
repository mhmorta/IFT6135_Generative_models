import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channels, latent_dim, cuda):
        super(Generator, self).__init__()
        self.cuda = cuda

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU()
            )

        self.convTranspose = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.25),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            # nn.Dropout2d(0.25),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.5),
            nn.ReLU(),

            nn.ConvTranspose2d(8, channels, 3, 1, 1),
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
            nn.Conv2d(channels, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(16, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            # nn.Dropout2d(0.5),
            # nn.LeakyReLU(0.2),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(0.2),            
            )
        self.mlp = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1)
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
