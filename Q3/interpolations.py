import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image

from gan.models import Generator
from vae.model import VAE

import argparse
import os


def make_interpolation(z, dim, eps=1):
    zh= z[0].clone().detach()
    zh[dim]= zh[dim]+ eps
    return zh


def save_images(Data, path, nrow=1):
    save_image(Data.data.view(-1,  3, 32, 32).cpu(), path, nrow, normalize=True)


def GAN_disentangled_representation_experiment(device):
    batch_size = 1
    latent_dim=100
    noise = Variable(torch.randn(batch_size, latent_dim)).to(device)

    G = Generator(channels=3, latent_dim=latent_dim, cuda=device).to(device)

    saved_model = './gan/results/models/gan_svhn_model.pt'
    G.load_state_dict(torch.load(saved_model, map_location=device), strict=False)

    dims = range(0,100)
    outputs = []
    z_y = G(noise).view(1, -1)
    for d in dims:
        zh = make_interpolation(noise, dim=d)
        output = Variable(G(zh)).to(device)
        outputs.append(output)

    outputs = torch.cat(outputs, dim=0).view(len(dims),-1)

    path = 'gan/results/interpolated/GAN_disentangled_zs.png'
    save_images(outputs, path)

    difference = torch.abs(outputs - z_y).view(100,-1)
    sum_dif = torch.sum(difference, dim=1).detach().cpu().numpy()
    top_sum_diff_indcs = np.unravel_index(np.argsort(sum_dif, axis=None), sum_dif.shape)[0]
    top_sum_diff_indcs = [top_sum_diff_indcs[x] for x in range(9, 100, 10)]
    print(top_sum_diff_indcs)
    
    top_k_images = Variable(outputs[top_sum_diff_indcs]).to(device).view(len(top_sum_diff_indcs), -1)
    top_k_images = torch.cat((z_y.view(1, -1), top_k_images))

    path = 'vae/results/interpolated/gan_top_disentangleds.png'
    save_images(top_k_images, path, nrow=len(top_k_images))



def  GAN_interpolating_experiment(device):
    batch_size = 2
    latent_dim = 100
    z = Variable(torch.randn(batch_size, latent_dim)).to(device)

    G = Generator(channels=3, latent_dim=latent_dim, cuda=device).to(device)

    saved_model = './gan/results/models/gan_svhn_model.pt'
    G.load_state_dict(torch.load(saved_model, map_location=device), strict=False)

    x = Variable(model.generate(z)).to(device)
    x_0 = x[0]
    x_1 = x[1]

    a_list = np.arange(0, 1, 0.1)
    z_list = []
    x_list = []
    for a in a_list:
        z_list.append(a*z[0] + (1-a)*z[1])
        x_list.append(a*x_0 + (1-a)*x_1)

    z_list = torch.cat(z_list, dim=0).view(len(a_list),-1)
    x_list = torch.cat(x_list, dim=0).view(len(a_list),-1)

    zh_y = Variable(G(z_list)).to(device)

    path = 'gan/results/interpolated/GAN_interpolated_zs.png'
    save_images(zh_y, path)

    path = 'gan/results/interpolated/GAN_interpolated_xs.png'
    save_images(x_list, path)

    path = 'vae/results/interpolated/GAN_interpolated_xs_zs.png'
    save_images(torch.cat((x_list, zh_y), dim=0), path, nrow=10)



def VAE_disentangled_representation_experiment(device):
    batch_size = 1
    latent_dim=100
    noise = Variable(torch.randn(batch_size, latent_dim)).to(device)

    model = VAE(100).to(device)

    saved_model = './vae/saved_model/params_epoch_24_loss_86.3193.pt'
    model.load_state_dict(torch.load(saved_model, map_location=device), strict=False)

    dims = range(0,100)
    outputs = []
    z_y =  Variable(model.generate(noise)).to(device)
    for d in dims:
        zh = make_interpolation(noise, dim=d).view(batch_size, latent_dim)
        output = Variable(model.generate(zh)).to(device)
        outputs.append(output)

    outputs = torch.cat(outputs, dim=0)

    path = 'vae/results/interpolated/vae_disentangled_zs.png'
    save_images(outputs, path)

    difference = torch.abs(outputs - z_y).view(100,-1)
    sum_dif = torch.sum(difference, dim=1).detach().cpu().numpy()
    top_sum_diff_indcs = np.unravel_index(np.argsort(sum_dif, axis=None), sum_dif.shape)[0]
    top_sum_diff_indcs = [top_sum_diff_indcs[x] for x in range(9, 100, 10)]
    print(top_sum_diff_indcs)
    
    top_k_images = Variable(outputs[top_sum_diff_indcs]).to(device).view(len(top_sum_diff_indcs), -1)
    top_k_images = torch.cat((z_y.view(1, -1), top_k_images))

    path = 'vae/results/interpolated/vae_top_disentangleds.png'
    save_images(top_k_images, path, nrow=len(top_k_images))

def VAE_interpolating_experiment(device):
    batch_size = 2
    latent_dim = 100
    z = Variable(torch.randn(batch_size, latent_dim)).to(device)

    model = VAE(100).to(device)

    saved_model = './vae/saved_model/params_epoch_24_loss_86.3193.pt'
    model.load_state_dict(torch.load(saved_model, map_location=device), strict=False)

    x = Variable(model.generate(z)).to(device)
    x_0 = x[0]
    x_1 = x[1]

    a_list = np.arange(0, 1, 0.1)
    z_list = []
    x_list = []
    for a in a_list:
        z_list.append(a*z[0] + (1-a)*z[1])
        x_list.append(a*x_0 + (1-a)*x_1)

    z_list = torch.cat(z_list, dim=0).view(len(a_list),-1)
    x_list =  torch.cat(x_list, dim=0).view(-1,3,32,32)

    zh_y = Variable(model.generate(z_list)).to(device)

    path = 'vae/results/interpolated/VAE_interpolated_zs.png'
    save_images(zh_y, path, nrow=len(zh_y))

    path = 'vae/results/interpolated/VAE_interpolated_xs.png'
    save_images(x_list, path, nrow= len(x_list))

    path = 'vae/results/interpolated/VAE_interpolated_xs_zs.png'
    save_images(torch.cat((x_list, zh_y), dim=0), path, nrow=10)

if __name__ == "__main__":
    # torch.manual_seed(511)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", type=str, default="GAN", help="Do the evaluation for either GAN or VAE")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    opt = parser.parse_args()

    # GAN_disentangled_representation_experiment(device)
    # GAN_interpolating_experiment(device)
    VAE_disentangled_representation_experiment(device)
    VAE_interpolating_experiment(device)

    print('Done...')


    # print(zh)
