import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad as torch_grad
import samplers as samplers

# a = np.randn.uniform(0, 1)
## the data also come form the distribution

cuda = torch.cuda.is_available();

X_dim = 2
h_dim = 64

a = np.random.uniform(0, 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.D = torch.nn.Sequential(
			    torch.nn.Linear(X_dim, h_dim),
			    torch.nn.ReLU(),
			    # torch.nn.Linear(X_dim, h_dim),
			    # torch.nn.ReLU(),
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

def JSD(D_x, D_y):
	D_loss_real = torch.mean(torch.log(D_x))
	D_loss_fake = torch.mean(torch.log(1-D_y))
	if cuda:
		D_loss_real = D_loss_real.cuda()
		D_loss_fake = D_loss_fake.cuda()

	# torch.log
	D_loss = torch.from_numpy(np.array([np.log(2)])).cuda().float() + 0.5 * (D_loss_real + D_loss_fake)
	return -D_loss

# WD = torch.mean()


def train_JSD():
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

			loss = JSD(O_real, O_fake)

			if ( e%10000 == True):
				print(-loss.data)

			loss.backward()
			optimizer.step()

		# testing the values
		O_real = Discriminator(X)
		O_fake = Discriminator(Y)

		loss = JSD(O_real, O_fake)

		print (-loss.data)
		losses.append(loss)
	# print(losses)
	print ('Done...')

def WD(D_x, D_y, X, Y):
	lam = 10
	D_loss_real = torch.mean(D_x)
	D_loss_fake = torch.mean(D_y)
	regularizer = gradient_penalty(X, Y)
	# print(regularizer)
	D_loss = (D_loss_real - D_loss_fake) - lam * (regularizer)
	return -D_loss

def gradient_penalty(X, Y):
	batch_size = X.size()[0]
	z = a * X + (1-a) * Y
	z = Variable(z, requires_grad= True)

	prob_z = Discriminator(z)

	# Calculate gradients of probabilities with respect to examples
	gradients = torch_grad(outputs=prob_z, inputs=z, grad_outputs=torch.ones(prob_z.size()).cuda() if cuda else torch.ones(
						   prob_z.size()),
    					   create_graph=True, retain_graph=True)[0]
	gradients = gradients.view(batch_size, -1)
	# gradients = gradients.norm(2, dim=1)
	gradients = torch.sqrt(gradients ** 2)
	gradients = gradients - 1
	gradients = gradients **2
	# print(gradients)
	return (gradients).mean()

def train_WD():
	losses = []
	thetas = np.array(range(-10, 11))/10
	for i in [0]:
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
		for e in range(10000):
			O_real = Discriminator(X)
			O_fake = Discriminator(Y)

			optimizer.zero_grad()

			loss = WD(O_real, O_fake, X, Y)

			if ( e%1000 == True):
				print(loss.data)

			loss.backward()
			optimizer.step()

		# testing the values
		O_real = Discriminator(X)
		O_fake = Discriminator(Y)

		loss = WD(O_real, O_fake, X, Y)

		print (loss.data)
		losses.append(loss)
	# print(losses)
	print ('Done...')



# train_JSD()
train_WD()


