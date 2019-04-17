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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        if m.bias is not None:
            m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.D(x)

if cuda:
	Discriminator = Net().cuda()
else:
	Discriminator = Net()

#  the criterion should be defined as it is asked in 1.1 and also 1.2, so two functions
# Discriminator loss


def WD(D_x, D_y, X, Y):
	lam = 10
	D_loss_real = torch.mean(D_x)
	D_loss_fake = torch.mean(D_y)
	# regularizer = gradient_penalty(X, Y)
	regularizer = _gradient_penalty(X, Y)
	# print(regularizer)
	D_loss = (D_loss_real - D_loss_fake) - lam * (regularizer)
	return -D_loss

def gradient_penalty(X, Y):
	batch_size = X.size()[0]
	a = torch.rand(512, 1)
	z = a * X + (1-a) * Y
	z = Variable(z, requires_grad= True)

	prob_z = Discriminator(z)

	# Calculate gradients of probabilities with respect to examples
	gradients = torch_grad(outputs=prob_z, inputs=z, grad_outputs=torch.ones(prob_z.size()).cuda() if cuda else torch.ones(
						   prob_z.size()),
    					   create_graph=True, retain_graph=True)[0]
	gradients = gradients.view(batch_size, -1)
	gradients = gradients.norm(2, dim=1)
	gradients = gradients - 1
	gradients = gradients **2
	# gradients = torch.sqrt(gradients ** 2)
	print(gradients)
	return (gradients).mean()

def _gradient_penalty( real_data, generated_data):
	batch_size = real_data.size()[0]

	# Calculate interpolation
	alpha = torch.rand(batch_size, 1)
	alpha = alpha.expand_as(real_data)
	if cuda:
		alpha = alpha.cuda()
	interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
	interpolated = Variable(interpolated, requires_grad=True)
	if cuda:
		interpolated = interpolated.cuda()

	# Calculate probability of interpolated examples
	prob_interpolated = Discriminator(interpolated)

	# Calculate gradients of probabilities with respect to examples
	gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
							grad_outputs=torch.ones(prob_interpolated.size()).cuda() if cuda else torch.ones(
							prob_interpolated.size()),
							create_graph=True, retain_graph=True)[0]

	# Gradients have shape (batch_size, num_channels, img_width, img_height),
	# so flatten to easily take norm per example in batch
	gradients = gradients.view(batch_size, -1)
	# self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

	# Derivatives of the gradient close to 0 can cause problems because of
	# the square root, so manually calculate norm and add epsilon
	gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

	# Return gradient penalty
	return ((gradients_norm - 1) ** 2).mean()

def train_WD():
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
		# print
		D_fake = next(samplers.distribution1(thetas[i], 512))

		X = torch.from_numpy(D_real).float()
		Y = torch.from_numpy(D_fake).float()

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
				print(-loss.data)

			loss.backward()
			optimizer.step()

		# testing the values
		O_real = Discriminator(X)
		O_fake = Discriminator(Y)

		loss = WD(O_real, O_fake, X, Y)

		print (-loss.data)
		losses.append(loss)
	# print(losses)
	print ('Done...')


train_WD()


