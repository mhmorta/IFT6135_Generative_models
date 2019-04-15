import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import samplers as samplers


## wasserstein distance setting
a = np.random.uniform(0, 1)
lam = 10


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

## the criterion should be defined as it is asked in 1.1 and also 1.2, so two functions
## Discriminator loss for the Jensen-shannon divergence

ones_label = Variable(torch.ones(512, 1)).cuda()
zeros_label = Variable(torch.zeros(512, 1)).cuda()
def JSD(D_x, D_y):
	D_loss_real = F.binary_cross_entropy(D_x, ones_label).cuda()
	D_loss_fake = F.binary_cross_entropy(D_y, zeros_label).cuda()
	D_loss = np.log(2) + 0.5 * (D_loss_real + D_loss_fake)
	return D_loss


# Discriminator loss for Wasserstain divergence
def WD(D_x, D_y, X, Y):
	D_loss_real = torch.mean(D_x)
	D_loss_fake = torch.mean(D_y)
	regularizer = gradient_penalty(X, Y)
	D_loss = (D_loss_real - D_loss_fake) + lam * (regularizer)
	return

def gradient_penalty(X, Y):
	batch_size = X.size()[0]
	z = a  * X + (1-a) * Y
	z = Variable(z, requires_grad= True)

	prob_z = self.Discriminator(z)

	# Calculate gradients of probabilities with respect to examples
	gradients = torch_grad(outputs=prob_z, inputs=z, grad_outputs=torch.ones(prob_z.size()).cuda() if self.use_cuda else torch.ones(prob_z.size()),
    	create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(gradients ** 2, dim=1)

    # Return gradient penalty
	return lam * ((gradients_norm - 1) ** 2).mean()

def train():
	# self.losses = {'G': [], 'D': [], 'GP': [],'gradient_norm': []}
	losses = []

	for i in range(1):
		D_fake = samplers.distribution1(thetas[0], 512)

		for e in range(100):
			X = Variable(torch.from_numpy(next(D_real)).float()).cuda()
			Y = Variable(torch.from_numpy(next(D_fake)).float()).cuda(


			O_real = Discriminator(X)
			O_fake = Discriminator(Y)
			
			loss_JSD = JSD(O_real,O_fake)
			# loss_WD = WD(O_real, O_fake, X, Y)

			losses.append(loss_JSD)
			loss_JSD.backward()

			# losses.append(loss_WD)
			# loss_WD.backward()

			optimizer.step()
			optimizer.zero_grad()
			# print (loss_WD)


train()



