import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad as torch_grad
import samplers as samplers

cuda = torch.cuda.is_available();
## wasserstein distance setting
a = np.random.uniform(0, 1)
lam = 10
losses = {'G': [], 'D': [], 'GP': [],'gradient_norm': []}

## the data also come form the distribution
thetas = np.array(range(-10, 11))/10
D_real = next(samplers.distribution1(0, 512))

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

## the criterion should be defined as it is asked in 1.1 and also 1.2, so two functions
## Discriminator loss for the Jensen-shannon divergence

ones_label = Variable(torch.ones(512, 1), requires_grad=False).cuda()
zeros_label = Variable(torch.zeros(512, 1), requires_grad=False).cuda()
def JSD(D_x, D_y):
	D_loss_real = F.binary_cross_entropy(D_x, ones_label).cuda()
	D_loss_fake = F.binary_cross_entropy(D_y, zeros_label).cuda()
	D_loss = np.log(2) + 0.5 * (D_loss_real + D_loss_fake)
	print
	return D_loss


# Discriminator loss for Wasserstain divergence
def WD(D_x, D_y, X, Y):
	D_loss_real = torch.mean(D_x)
	D_loss_fake = torch.mean(D_y)
	regularizer = gradient_penalty(X, Y)
	D_loss = (D_loss_real - D_loss_fake) - lam * (regularizer)
	return D_loss

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
	return ((gradients.norm(2, dim=1) -1)**2).mean()

def train():
	

	for i in range(21):
		if cuda:
			Discriminator = Net().cuda()
		else:
			Discriminator = Net()
		X = torch.from_numpy(D_real).float()
		Y = torch.from_numpy(D_real).float()
		Y[:, 0] = thetas[i]

		if cuda:
			X = X.cuda()
			Y = Y.cuda()

		for e in range(50000):
			O_real = Discriminator(X)
			O_fake = Discriminator(Y)
			optimizer.zero_grad()

			loss_WD = WD(O_real, O_fake, X, Y)

			if (e%10000 ==True):
				print (loss_WD.data)

			loss_WD.backward()

			optimizer.step()
			losses['D'].append(loss_WD)




train()



