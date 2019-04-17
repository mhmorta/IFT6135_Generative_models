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
h_dim = 32

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
        # if m.bias is not None:
        #     m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.D(x)


class D(nn.Module):
  def __init__(self, input_dim=1, hidden_size=32, n_hidden=3, with_activation=False):
    super(D, self).__init__()

    modules= [ nn.Linear(input_dim, hidden_size) ,  nn.ReLU() ]
    for i in range(n_hidden - 1):
      modules.append(nn.Linear(hidden_size, hidden_size) )
      modules.append(nn.ReLU())
						
    modules.append(nn.Linear(hidden_size, 1) )
    
    if with_activation:
      modules.append(nn.Sigmoid())

    self.net = nn.Sequential(*modules)
    self.net.apply(self.init_weights)

    self.input_dim = input_dim

  def init_weights(self, m):
    if type(m) == nn.Linear:
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        m.bias.data.fill_(0.0)

  def forward(self, input):
    return self.net(input)


# if cuda:
# 	# Discriminator = Net().cuda()
# 	Discriminator = D(input_dim=2, hidden_size=32, n_hidden=3, with_activation=False).cuda()
# else:
# 	# Discriminator = Net()
# 	Discriminator = D(input_dim=2, hidden_size=32, n_hidden=3, with_activation=False)


#  the criterion should be defined as it is asked in 1.1 and also 1.2, so two functions
# Discriminator loss


# def WD(D_x, D_y, X, Y):
# 	lam = 10
# 	D_loss_real = torch.mean(D_x)
# 	D_loss_fake = torch.mean(D_y)
# 	regularizer = gradient_penalty(X, Y)
# 	# regularizer = _gradient_penalty(X, Y)
# 	# regularizer = computeGP(X, Y)
# 	# regularizer = (torch.mean(((regularizer - 1)**2)))
# 	# print(regularizer)
# 	regularizer = lam * (regularizer)
# 	D_loss = -(D_loss_real - D_loss_fake - (regularizer))
# 	return D_loss

def WD2(Discriminator, X, Y):
	lam = 10
	D_loss_real = torch.mean(Discriminator(X))
	D_loss_fake = torch.mean(Discriminator(Y))
	regularizer = gradient_penalty(Discriminator, X, Y)
	regularizer = lam * (regularizer)
	D_loss = -(D_loss_real - D_loss_fake - (regularizer))
	return D_loss

def computeGP(Discriminator,  p, q):
  size = p.shape[0]
  a = torch.empty(size,1).uniform_(0,1).cuda()
  z = a * p + (1 - a) * q
  z.requires_grad = True
  out = Discriminator(z)

  gradients = torch_grad(outputs=out, inputs=z, 
                   grad_outputs=torch.ones(out.size()).cuda() if cuda else torch.ones(out.size()),
                   create_graph=True, only_inputs=True, 
                   retain_graph=True)[0]
  gradients = gradients.view(gradients.size(0), -1)
  gradients_norm = gradients.norm(2, dim=1)

  return gradients_norm

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
	gradients = gradients.norm(2, dim=1)
	gradients = gradients - 1
	gradients = gradients **2
	# gradients = torch.sqrt(gradients ** 2)
	# print(gradients)
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
	p = samplers.distribution1(0, 512)


	#optim = torch.optim.SGD(model.parameters(), lr=0.001)
	# optim = torch.optim.Adam(Discriminator.parameters(), lr=0.001)

	for i in range(21):
		if cuda:
			Discriminator = Net().cuda()
		else:
			Discriminator = Net()

		p = samplers.distribution1(0, 512)
		q = samplers.distribution1(thetas[i], 512)
		dist_p = iter(p)
		dist_q = iter(q)		
		optimizer = optim.Adam(Discriminator.parameters(), lr = 1e-3)

		print(thetas[i])
		# print

		
		#  training stage
		for e in range(3000):
			Discriminator.train()
			X = torch.from_numpy(next(dist_p)).float()
			Y = torch.from_numpy(next(dist_q)).float()

			if cuda:
				X = X.cuda()
				Y = Y.cuda()

			# p_tensor = Variable( torch.from_numpy(np.float32(next(dist_p).reshape(512, 2))) )
			# q_tensor = Variable( torch.from_numpy(np.float32(next(dist_p).reshape(512, 2))) )

			# if cuda:
			# 	p_tensor = p_tensor.cuda()
			# 	q_tensor = q_tensor.cuda()
			optimizer.zero_grad()
			# O_real = Discriminator(X)
			# O_fake = Discriminator(Y)


			loss = WD2(Discriminator, X,Y)
			# loss = WD2(Discriminator, p_tensor, q_tensor )

			# if ( e%1000 == True):
			# 	print(-loss.data)

			loss.backward()
			optimizer.step()

		# testing the values
		Discriminator.eval()


		X = torch.from_numpy(next(dist_p)).float()
		Y = torch.from_numpy(next(dist_q)).float()

		if cuda:
			X = X.cuda()
			Y = Y.cuda()

		loss = WD2(Discriminator, X,Y)

		print (-loss.data)
		losses.append(loss)
	# print(losses)
	print ('Done...')

def train(p=None, q=None, batch_size=512, epochs=1000, log=True):
  if cuda:
	# Disc  riminator = Net().cuda()
    Discriminator = D(input_dim=2, hidden_size=32, n_hidden=3, with_activation=False).cuda()
  else:
	# Discriminator = Net()
    Discriminator = D(input_dim=2, hidden_size=32, n_hidden=3, with_activation=False)

  Discriminator.train()
  batch_size = 512
  p = samplers.distribution1(0, batch_size)
  q = samplers.distribution1(-0.5, batch_size)
  #optim = torch.optim.SGD(model.parameters(), lr=0.001)
  optim = torch.optim.Adam(Discriminator.parameters(), lr=0.001)
  dist_p = iter(p)
  dist_q = iter(q)

  for e in range(epochs):
    optim.zero_grad()

    # p data.
    px = next(dist_p)

    # q data.
    qx = next(dist_q)

    p_tensor = Variable( torch.from_numpy(np.float32(px.reshape(batch_size, 2))) )
    q_tensor = Variable( torch.from_numpy(np.float32(qx.reshape(batch_size, 2))) )

    if cuda:
        p_tensor = p_tensor.cuda()
        q_tensor = q_tensor.cuda()

    loss = WD2(Discriminator, p_tensor, q_tensor)
    loss.backward()
    optim.step()


    if log:
      if e % 100 == 0 or (e < 100 and e % 10 == 0):
        print("\tEpoch ", e, "Loss = ", loss.data.cpu().numpy())


# train()
train_WD()


