import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import samplers as samplers


## the data also come form the distribution
cuda = torch.cuda.is_available();

f_0 = next(samplers.distribution3(512))
f_1 = next(samplers.distribution4(1))

X_dim = 1
h_dim = 64

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.D = torch.nn.Sequential(
			    torch.nn.Linear(X_dim, h_dim),
			    torch.nn.ReLU(),
			    torch.nn.Linear(h_dim, h_dim),
			    torch.nn.ReLU(),
			    torch.nn.Linear(h_dim, 1),
			    torch.nn.Sigmoid()
			)
    def forward(self, x):
        return self.D(x)

if cuda:
	Discriminator = Net().cuda()
else:
	Discriminator = Net()

optimizer = optim.SGD(Discriminator.parameters(), lr = 1e-3, momentum = 0.9)

def GAN(D_x, D_y):
	D_loss_real = torch.mean(torch.log(D_x))
	D_loss_fake = torch.mean(torch.log(1 - D_y))
	D_loss = D_loss_real + D_loss_fake
	return - D_loss



def train():
	losses = []

	for e in range(100000):
		X = torch.from_numpy(f_0).float()
		Y = torch.from_numpy(f_1).float()

		if cuda:
			X = X.cuda()
			Y = Y.cuda()

		O_real = Discriminator(X)
		O_fake = Discriminator(Y)
		optimizer.zero_grad()
		
		loss = GAN(O_real,O_fake)

		if (e%1000 ==True):
			print (loss.data)

		loss.backward()
		optimizer.step()

		losses.append(loss)
		

train()



