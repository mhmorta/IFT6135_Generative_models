import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image

from models import Generator

import argparse
import os


def make_interpolation(z, dim, eps=0.1):
    zh= z[0].clone().detach()
    zh[dim]= zh[dim]+ eps
    return zh


def save_images(Data, path):
    save_image(Data.data.view(-1, 3, 32, 32).cpu(), path, nrow = 1, normalize=True)


def gan_disentangled_representation_experiment(device):
    batch_size = 1
    latent_dim=100
    noise = Variable(torch.randn(batch_size, latent_dim)).to(device)

    G = Generator(channels=3, latent_dim=latent_dim, cuda=device).to(device)

    saved_model = './results/models/gen_svhn_model.pt'
    G.load_state_dict(torch.load(saved_model, map_location=device), strict=False)

    dims = [0,20]
    outputs = []
    for d in dims:
        zh = make_interpolation(noise, dim=d)
        output = Variable(G(noise)).to(device)
        outputs.append(output)

    outputs = torch.cat(outputs, dim=0).view(len(dims),-1)

    path = 'results/interpolated/gan_disentangled_zs.png'
    save_images(outputs, path)



def gan_interpolating_experiment(device):
    batch_size = 2
    latent_dim = 100
    z = Variable(torch.randn(batch_size, latent_dim)).to(device)

    G = Generator(channels=3, latent_dim=latent_dim, cuda=device).to(device)

    saved_model = './results/models/gen_svhn_model.pt'
    G.load_state_dict(torch.load(saved_model, map_location=device), strict=False)

    x_0 = Variable(G(z[0])).to(device)
    x_1 = Variable(G(z[1])).to(device)

    a_list = np.arange(0, 1, 0.1)
    z_list = []
    x_list = []
    for a in a_list:
        z_list.append(a*z[0] + (1-a)*z[1])
        x_list.append(a*x_0 + (1-a)*x_0)

    z_list = torch.cat(z_list, dim=0).view(len(a_list),-1)
    x_list =  torch.cat(x_list, dim=0).view(len(a_list),-1)

    zh_y = Variable(G(z_list)).to(device)

    path = 'results/interpolated/gan_interpolated_zs.png'
    save_images(zh_y, path)

    path = 'results/interpolated/gan_interpolated_xs.png'
    save_images(x_list, path)


if __name__ == "__main__":
    torch.manual_seed(5)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", type=str, default="GAN", help="Do the evaluation for either GAN or VAE")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    opt = parser.parse_args()

    if opt.evaluate == "GAN":
        gan_interpolating_experiment(device)


    # print(zh)
