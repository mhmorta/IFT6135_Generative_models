import argparse
import os
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torchvision.utils import save_image
import samplers as samplers

from models import Generator, Discriminator
from data_loaders import get_data_loader

def sample_generator(Generator, num_samples, latent_dim, update_d, device):
    noise = Variable(torch.randn(num_samples, latent_dim), requires_grad=False).to(device)
    noise.require_grad = False

    gen_samples = Generator(noise)
    gen_samples = gen_samples.view(-1, 3, 32, 32)
    save_image(gen_samples.data.view(num_samples, 3, 32, 32).cpu(), 'results/gs' + str(update_d) + '.png', nrow = 10)


def loss_WD(Discriminator, D_x, D_y, batch_size, device):
    lam = 10
    D_loss_real = Discriminator(D_x.detach())
    D_loss_fake = Discriminator(D_y)
    regularizer = gradient_penalty(Discriminator, D_x, D_y, batch_size, device)
    D_loss = (D_loss_real.mean() - D_loss_fake.mean()) - (lam * regularizer)
    return -D_loss 


def gradient_penalty(Discriminator, p, q, batch_size, device):
    a = torch.randn(batch_size, 1, device=device, requires_grad = True)
    a_total = a.expand(batch_size, p.nelement()//batch_size).contiguous()
    a_total = a_total.view(p.size())

    z = a_total * p + (1 - a_total) * q
    z_prob = Discriminator(z)

    gradients = torch_grad(z_prob.mean(), z, create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gardient_pen = ((gradients.norm(2, dim=1) -1)**2).mean()
    return gardient_pen


def train(Discriminator, Generator, trainloader, latent_dim, batch_size, epochs, device):
    Discriminator.train()
    Generator.train()

    # ## optimizers
    optimizer_G = torch.optim.Adam(Generator.parameters(), lr=opt.lr*2, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for e in range(epochs):
        for i, (img, _) in enumerate(trainloader):
            batch_size = img.shape[0]
            update_d = e * batch_size + i + 1
            D_x = Variable(img.to(device))

            noise = Variable(torch.randn(batch_size, latent_dim)).to(device)
            D_y = Variable(Generator(noise)).to(device)
            # if cuda:
            #     D_x = D_x.cuda()
            #     D_y = D_y.cuda()
            #     noise = noise.cuda()

            loss_d = loss_WD(Discriminator, D_x, D_y, batch_size, device)
            Discriminator.zero_grad()
            loss_d.backward()
            optimizer_D.step()


            if update_d % 5 == 0:
                noise = Variable(torch.randn(batch_size, latent_dim)).to(device)
                D_y = Variable(Generator(noise)).to(device)

                # if cuda:
                #     D_y = D_y.cuda()
                #     noise = noise.cuda()
                D_loss_fake = Discriminator(D_y)
                loss_g = -(D_loss_fake.mean())

                Generator.zero_grad()
                Discriminator.zero_grad()
                loss_g.backward()
                optimizer_G.step()

        print ('Epoch: ', e, 'step: ', i, 'Discriminator loss: ', loss_d.mean().cpu().data.numpy(),
            'Generator loss: ', loss_g.mean().cpu().data.numpy())

        sample_generator(Generator, 100, latent_dim, update_d, device)


if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda = torch.cuda.is_available()
    # print (torch.cuda.current_device())

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
    parser.add_argument("--optimizer", type=str, default='Adam', help="type of the optimizer")
    parser.add_argument("--lr", type=float, default=1e-9, help="adam: learning rate")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    parser.add_argument("--lamda", type=int, default=10, help='the lambda value for gradient penalty')
    opt = parser.parse_args()

    device = torch.device("cuda" if cuda else "cpu")

    D = Discriminator(opt.channels, opt.latent_dim, device)
    G = Generator(opt.channels, opt.latent_dim, device)
    
    print(D)
    print(G)
    
    D.to(device)
    G.to(device)

    trainloader, validloader, testloader = get_data_loader('./data', 512)
    train(D, G, trainloader, opt.latent_dim, opt.batch_size, opt.epochs, device)

    name = 'svhn_model'
    torch.save(G.state_dict(), './results/models/gen_' + name + '.pt')
    torch.save(D.state_dict(), './results/models/dis_' + name + '.pt')
