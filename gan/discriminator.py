import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import samplers as samplers

# a = np.randn.uniform(0, 1)
## the data also come form the distribution

thetas = np.array(range(-10, 11))/10
D_real = samplers.distribution1(0, 512)

X_dim = 2
h_dim = 64

D = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
    torch.nn.Sigmoid()
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.D = torch.nn.Sequential(
			    torch.nn.Linear(X_dim, h_dim),
			    torch.nn.ReLU(),
			    torch.nn.Linear(h_dim, 1),
			    torch.nn.Sigmoid()
			)
    def forward(self, x):
        return self.D(x)

Discriminator = Net().cuda()



optimizer = optim.SGD(Discriminator.parameters(), lr = 1e-3, momentum = 0.9)

#  the criterion should be defined as it is asked in 1.1 and also 1.2, so two functions
# Discriminator loss

ones_label = Variable(torch.ones(512, 1)).cuda()
zeros_label = Variable(torch.zeros(512, 1)).cuda()
def JSD(D_x, D_y):
	D_loss_real = F.binary_cross_entropy(D_x, ones_label).cuda()
	D_loss_fake = F.binary_cross_entropy(D_y, zeros_label).cuda()
	D_loss = np.log(2) + 0.5 * (D_loss_real + D_loss_fake)
	return D_loss

# WD = torch.mean()


def train():
	losses = []

	for i in range(1):
		D_fake = samplers.distribution1(thetas[10], 512)

		for e in range(100):
			X = Variable(torch.from_numpy(next(D_real)).float()).cuda()
			Y = Variable(torch.from_numpy(next(D_fake)).float()).cuda()

			O_real = Discriminator(X)
			O_fake = Discriminator(Y)
			optimizer.zero_grad()

			loss = JSD(O_real,O_fake)

			losses.append(loss)
 	 		
			loss.backward()
			optimizer.step()
			print (loss)


train()



