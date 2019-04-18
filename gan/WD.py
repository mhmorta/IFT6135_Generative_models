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

class Net(nn.Module):
    def __init__(self, X_dim=2, h_dim= 32):
        super(Net, self).__init__()
        self.X_dim = X_dim
        self.h_dim = h_dim
        self.D = torch.nn.Sequential(
			    torch.nn.Linear(X_dim, h_dim),
			    torch.nn.ReLU(),
			    torch.nn.Linear(h_dim, h_dim),
			    torch.nn.ReLU(),
			    torch.nn.Linear(h_dim, 1),
			)
        self.D.apply(self.init_weights)	

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.D(x)

def loss_WD(Discriminator, X, Y):
	lam = 10
	D_loss_real = torch.mean(Discriminator(X))
	D_loss_fake = torch.mean(Discriminator(Y))
	regularizer = gradient_penalty(Discriminator, X, Y)
	regularizer = lam * (regularizer)
	D_loss = -(D_loss_real - D_loss_fake - (regularizer))
	return D_loss

def gradient_penalty(Discriminator, X, Y):
	batch_size = X.size()[0]
	a = torch.empty(512,1).uniform_(0,1).cuda()
	z = a * X + (1-a) * Y
	z = Variable(z, requires_grad= True)

	prob_z = Discriminator(z)

	# Calculate gradients of probabilities with respect to examples
	gradients = torch_grad(outputs=prob_z, inputs=z, 
							grad_outputs=torch.ones(prob_z.size()).cuda() if cuda else torch.ones(prob_z.size()),
		                   create_graph=True, only_inputs=True, 
    					   retain_graph=True)[0]
	gradients = gradients.view(batch_size, -1)
	return ((gradients.norm(2, dim=1)-1) **2).mean()


## ----------------
## Training 
## ----------------


def train_WD():
	losses = []
	thetas = np.array(range(-10, 11))/10
	p = samplers.distribution1(0, 512)

	for i in range(len(thetas)):
		if cuda:
			Discriminator = Net().cuda()
		else:
			Discriminator = Net()

		p = samplers.distribution1(0, 512)
		q = samplers.distribution1(thetas[i], 512)
		dist_p = iter(p)
		dist_q = iter(q)		
		optimizer = optim.Adam(Discriminator.parameters(), lr = 1e-3)

		print('theta:', thetas[i])
		
		#  training stage
		for e in range(5000):
			Discriminator.train()
			X = torch.from_numpy(next(dist_p)).float()
			Y = torch.from_numpy(next(dist_q)).float()

			if cuda:
				X = X.cuda()
				Y = Y.cuda()

			optimizer.zero_grad()

			loss = loss_WD(Discriminator, X,Y)
			if ( e%1000 == True):
				print(-loss.data)

			loss.backward()
			optimizer.step()

		# testing the values
		Discriminator.eval()

		X = torch.from_numpy(next(dist_p)).float()
		Y = torch.from_numpy(next(dist_q)).float()

		if cuda:
			X = X.cuda()
			Y = Y.cuda()

		loss = loss_WD(Discriminator, X,Y)

		print (-loss.data)
		losses.append(-loss)
	print(losses)
	print ('Done...')

	plt.figure()
	plt.plot(thetas,losses)
	plt.title('Wasserstein GAN')
	plt.savefig('Wasserstein_D.png')
	plt.close()

train_WD()


