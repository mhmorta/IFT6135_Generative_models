import torch
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

def save_interpolated_image(D_y,dim):
    gen_samples = D_y[0].view(-1, 3, 32, 32)
    path = 'results/interpolated/'+ str(dim) + '.png'
    save_image(gen_samples.data.view(-1, 3, 32, 32).cpu(), path, nrow = 1, normalize=True)


if __name__ == "__main__":
    torch.manual_seed(5)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    opt = parser.parse_args()

    batch_size = 1
    noise = Variable(torch.randn(1, opt.latent_dim)).to(device)

    G = Generator(channels=3, latent_dim=100, cuda=device).to(device)

    saved_model = './results/models/gen_svhn_model.pt'
    G.load_state_dict(torch.load(saved_model, map_location=device), strict=False)

    dims = [0,20]
    for d in dims:
        zh = make_interpolation(noise, dim=d)
        D_y = Variable(G(noise)).to(device)
        save_interpolated_image(D_y, d)

    



    # print(zh)
