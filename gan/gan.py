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


class Discriminator(nn.Module):
  def __init__(self, input_dim=1, hidden_size=32, n_hidden=3, with_activation=True):
    super(Discriminator, self).__init__()

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



def train(model, p, q, loss_func, batch_size=512, epochs=1000):
  model.train()

  #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  dist_p = iter(p)
  dist_q = iter(q)

  for e in range(epochs):
    optimizer.zero_grad()

    px = next(dist_p)

    # q data.
    qx = next(dist_q)

    p_tensor = Variable( torch.from_numpy(np.float32(px.reshape(batch_size, model.input_dim))))
    q_tensor = Variable( torch.from_numpy(np.float32(qx.reshape(batch_size, model.input_dim))))

    D_x = model(p_tensor)
    D_y = model(q_tensor)

    loss = loss_func(D_x, D_y)

    loss.backward()
    optimizer.step()


    if e % 100 == 0:
        print("\tEpoch ", e, "Loss = ", loss.data.numpy())


def test_net(model, loss_fn, p, q, batch_size):
  px = next(iter(p))
  qx = next(iter(q))
  p_tensor = Variable( torch.from_numpy(np.float32(px.reshape(batch_size, model.input_dim))) )
  q_tensor = Variable( torch.from_numpy(np.float32(qx.reshape(batch_size, model.input_dim))) )

  return loss_fn(model, p_tensor, q_tensor, lambda_fact=0)
  
		
def gan_eval():
  epochs = 1000
  batch_size=1000
  hidden_size = 50
  n_hidden = 3
  input_dim = 1

  f0 = samplers.distributionGaussian(batch_size)
  f1 = samplers.distribution4(batch_size)

  D = Discriminator(input_dim, hidden_size, n_hidden)
  train(D, f0, f1, GAN, batch_size=batch_size, epochs=epochs)


  xx = torch.randn(10000)
  f = lambda x: torch.tanh(x*2+1) + x*0.75
  d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
  plt.hist(f(xx), 100, alpha=0.5, density=1)
  plt.hist(xx, 100, alpha=0.5, density=1)
  plt.xlim(-5,5)
  plt.savefig('histogram')
  plt.close()
  

  
  xx = np.linspace(-5,5,1000)
  N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
  f0_x_tensor = Variable( torch.from_numpy(np.float32(xx.reshape(batch_size, input_dim))) )
  D_x = D(f0_x_tensor)
  f1_est = N(f0_x_tensor) * D_x / (1 - D_x)


  # Plot the discriminator output.
  r = D_x.detach().numpy() 
  plt.figure(figsize=(8,4))
  plt.subplot(1,2,1)
  plt.plot(xx,r)
  plt.title(r'$D(x)$')

  estimate = f1_est.detach().numpy() 

  # Plot the density.
  plt.subplot(1,2,2)
  plt.plot(xx,estimate)
  plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
  plt.legend(['Estimated','True'])
  plt.title('Estimated vs True')
  plt.savefig('Estimated_vs_Exact.png') 
  plt.close()


gan_eval()



