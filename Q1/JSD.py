import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad as torch_grad
import samplers as samplers



cuda = torch.cuda.is_available();

X_dim = 2
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

# the criterion should be defined as it is asked in 1.1
# Discriminator loss

def JSD(D_x, D_y):
	D_loss_real = torch.mean(torch.log(D_x))
	D_loss_fake = torch.mean(torch.log(1-D_y))
	if cuda:
		D_loss_real = D_loss_real.cuda()
		D_loss_fake = D_loss_fake.cuda()
	D_loss = torch.from_numpy(np.array([np.log(2)])).cuda().float() + 0.5 * (D_loss_real + D_loss_fake)
	return -D_loss

## ----------------
## Training 
## ----------------


def train_JSD():
	losses = []
	thetas = np.array(range(-10, 11))/10
	D_real = next(samplers.distribution1(0, 512))
	for i in range(21):
		if cuda:
			Discriminator = Net().cuda()
		else:
			Discriminator = Net()

		# optimizer = optim.SGD(Discriminator.parameters(), lr = 1e-3, momentum = 0.9)
		optimizer = optim.Adam(Discriminator.parameters(), lr = 1e-3)

		print(thetas[i])
		
		D_fake = next(samplers.distribution1(thetas[i], 512))
		# print

		X = torch.from_numpy(D_real).float()
		Y = torch.from_numpy(D_fake).float()

		if cuda:
			X = X.cuda()
			Y = Y.cuda()
		
		#  training stage
		for e in range(1000):
			O_real = Discriminator(X)
			O_fake = Discriminator(Y)

			optimizer.zero_grad()

			loss = JSD(O_real, O_fake)

			if ( e%100 == True):

				print(-loss.data)

			loss.backward()
			optimizer.step()

		# testing the values
		O_real = Discriminator(X)
		O_fake = Discriminator(Y)

		loss = JSD(O_real, O_fake)

		print (-loss.data)
		losses.append(-loss)
	print ('Done...')

	losses = np.array(losses)
	plt.figure()
	plt.scatter(thetas,losses)
	plt.title('Jenssen Shanon Divergence')
	plt.xlabel('Theta')
	plt.ylabel('Divergance')
	plt.savefig('Jenssen_Shanon_Divergence.png') 
	plt.close()

train_JSD()






