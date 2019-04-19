import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import samplers as samplers

from data_loader import get_data_loader
from models import Generator, Discriminator
from Trainer import Trainer

cuda = torch.cuda.is_available()
cuda = False

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--optimizer", type=str, default='SGD', help="type of the optimizer")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--lambda_gp", type=int, default=10, help='the lambda value for gradient penalty')
args = parser.parse_args()




trainloader, validloader, testloader = get_data_loader('./data', 512)
data_loaders = {"trainloader": trainloader,
               "validloader": validloader,
                "testloader": testloader
                }

img_size = (32, 32, 1)

generator = Generator(args.latent_dim)
discriminator = Discriminator(args.channels, args.latent_dim)

## optimizers
if args.optimizer == "ADAM":
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
else:
    gen_optimizer = torch.optim.SGD(generator.parameters(), lr=args.lr)
    dis_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.lr)

# Train model
trainer = Trainer(generator, discriminator, gen_optimizer= gen_optimizer, dis_optimizer= dis_optimizer,
            use_cuda=cuda, batch_size = args.batch_size, data_loaders=data_loaders, epochs =args.epochs)
trainer.train()

## training
