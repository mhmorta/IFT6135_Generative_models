import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import samplers 

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, data_loaders,
                 gp_weight=10, critic_iterations=5, print_every=50, lambda_gp=10, epochs = 50, 
                 batch_size = 512, use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.lambda_gp = lambda_gp
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_loaders = data_loaders

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def loss_WD(self, X, Y):
        D_loss_real = torch.mean(self.D(X))
        D_loss_fake = torch.mean(self.D(Y))
        regularizer = self.gradient_penalty( X, Y)
        regularizer = self.lambda_gp * (regularizer)
        D_loss = -(D_loss_real - D_loss_fake - (regularizer))
        return D_loss
    
    def g_loss(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        return g_loss
  
    def gradient_penalty(self, X,Y):
        batch_size = X.size()[0]
        a = np.random.uniform(0, 1)
        z = a * X + (1-a) * Y
        Z = Variable(z, require_grad=True)

        prob_z = self.D(Z)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_z, inputs=z, grad_outputs=torch.ones(prob_z.size()).cuda() if cuda else torch.ones(
                            prob_z.size()),
                            create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        return ((gradients.norm(2, dim=1) -1)**2).mean()

    def train(self):
        trainloader = self.data_loaders["trainloader"]
        for epoch in range(self.epochs):
            for i,(imgs, _) in enumerate(trainloader):

                real = Variable(torch.ones((imgs.size(0), 1)), requires_grad=False)
                fake = Variable(torch.ones((imgs.size(0), 1)), requires_grad=False)

                real_imgs = Variable(imgs.type(torch.Tensor))

                # -----------------
                ## Train generator
                # -----------------

                self.G_opt.zero_grad()

                # noise = Variable(torch.Tensor(next(samplers.distribution3(self.batch_size))))
                noise = Variable(torch.randn(100, 100))

                ## generate a batch of images
                gen_imgs = self.G(noise)

                ## loss of generator (what is the adversarial loss)
                g_loss = self.g_loss(gen_imgs)

                g_loss.backward()
                self.G_opt.step()

                # ---------------------
                ## Train discriminator
                # ---------------------

                self.D_opt.zero_grad()

                ## loss of discriminator
                d_loss = self.loss_WD(self.D(real_imgs), self.D(gen_imgs))

                d_loss.backward()
                self.D_opt.step()


                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.epochs, i, len(trainloader), d_loss.item(), g_loss.item())
                    )
                
                # batches_done = epoch * len(trainloader) + i
                # if batches_done % opt.sample_interval == 0:
                #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data