import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import samplers as samplers

# a = np.randn.uniform(0, 1)
## the data also come form the distribution

cuda = torch.cuda.is_available();

# D_real = next(samplers.distribution1(0, 512))

X_dim = 2
h_dim = 64

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

if cuda:
	Discriminator = Net().cuda()
else:
	Discriminator = Net()

optimizer = optim.SGD(Discriminator.parameters(), lr = 1e-3, momentum = 0.9)

#  the criterion should be defined as it is asked in 1.1 and also 1.2, so two functions
# Discriminator loss

# def move_to_cuda(arg):
# 	if torch.cuda.is_available():
# 		return arg.cuda()
# 	return arg


ones_label = torch.ones(512, 1)
zeros_label = torch.zeros(512, 1)
if cuda:
	ones_label = ones_label.cuda()
	zeros_label = zeros_label.cuda()
def JSD(D_x, x_target, D_y, y_target):
	
	D_loss_real = torch.mean(torch.log(D_x))
	D_loss_fake = torch.mean(torch.log(1-D_y))
	if cuda:
		D_loss_real = D_loss_real.cuda()
		D_loss_fake = D_loss_fake.cuda()

	# torch.log
	D_loss = torch.from_numpy(np.array([np.log(2)])).cuda().float() + 0.5 * (D_loss_real + D_loss_fake)
	return -D_loss

# WD = torch.mean()


def train():
	losses = []
	thetas = np.array(range(-10, 11))/10
	for i in range(21):
		if cuda:
			Discriminator = Net().cuda()
		else:
			Discriminator = Net()

		optimizer = optim.SGD(Discriminator.parameters(), lr = 1e-3, momentum = 0.9)

		print(thetas[i])
		D_real = next(samplers.distribution1(0, 512))
		# print

		X = torch.from_numpy(D_real).float()
		Y = torch.from_numpy(D_real).float()
		Y[:,0] = thetas[i]
		if cuda:
			X = X.cuda()
			Y = Y.cuda()
		
		#  training stage
		for e in range(50000):
			O_real = Discriminator(X)
			O_fake = Discriminator(Y)

			optimizer.zero_grad()

			if (thetas[i] != 0):
				loss = JSD(O_real, ones_label, O_fake, zeros_label)
			else:
				loss = JSD(O_real, ones_label, O_fake, ones_label)

			if ( e%5000 == True):
				print(loss)

			loss.backward()
			optimizer.step()

		# testing the values
		O_real = Discriminator(X)
		O_fake = Discriminator(Y)

		if (thetas[i] != 0):
			loss = JSD(O_real, ones_label, O_fake, zeros_label)
		else:
			loss = JSD(O_real, ones_label, O_fake, ones_label)
		print (-loss.data)
		losses.append(loss)
	# print(losses)
	print ('Done...')




train()

# X = torch.ones((512,1)).cuda()
# Y = torch.zeros((512,1)).cuda()
# print(JSD(X,Y))



