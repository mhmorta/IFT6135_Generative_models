import argparse
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import dataset
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import samplers as samplers



cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--optimizer", type=str, default='SGD', help="type of the optimizer")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--lamda", type=int, default=10, help='the lambda value for gradient penalty')
opt = parser.parse_args()

def get_data_loader(dataset_location, batch_size):

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))])


    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader


trainloader, validloader, testloader = get_data_loader('./data', 512)

## Generator
class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequntial(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 512, 2),
            )
        self.mlp = nn.Sequntial(
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(512,10),
            )
    def forward(self, x):
        return self.mlp(self.model(x)[:,:,0, 0])

## Discriminator
class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequntial(
            nn.Linear(2, opt.latent_dim),
            nn.ReLU(),
            nn.Linear(opt.latent_dim, opt.latent_dim),
            nn.ReLU(),
            nn.Linear(opt.latent_dim, 1),
            nn.Sigmoid(),
            )
    def forward(self, x):
        return self.model(x)


def WD(D_x, D_y, X, Y):
    lam = opt.lamda
    D_loss_real = torch.mean(D_x)
    D_loss_fake = torch.mean(D_y)
    D_loss = D_loss_real - D_loss_fake - (lam * gradient_penalty)
    return -D_loss
  
def gradient_penalty(X,Y):
    batch_size = X.size()[0]
    a = np.random.uniform(0, 1)
    z = a * X + (1-a) * Y
    Z = Variable(z, require_grad=True)

    prob_z = Discriminator(Z)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_z, inputs=z, grad_outputs=torch.ones(prob_z.size()).cuda() if cuda else torch.ones(
                           prob_z.size()),
                           create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    gardient_pen = ((gradients.norm(2, dim=1) -1)**2).mean()
    return gardient_pen


## Initializing the generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda():
    generator.cuda()
    discriminator.cuda()


## optimizers
optimizer_G = torch.optim.opt.optimizer(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.opt.optimizer(discriminator.parameters(), lr=opt.lr)


## training
for epoch in range(opt.epoch):
    for i,(imgs, _) in enumerate(trainloader):

        real = Variable(Tensor(imgs.size(0), 1).fill_(1.0), require_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(1.0), require_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        ## Train generator
        # -----------------

        optimizer_G.zero_grad()

        noise = Variable(Tensor(samplers.distribution1(0, opt.batch_size)))

        ## generate a batch of images
        gen_imgs = generator(noise)

        ## loss of generator (what is the adversarial loss)
        g_loss = ????

        g_loss.backward()
        optimizer_G.step()


        # ---------------------
        ## Train discriminator
        # ---------------------

        optimizer_D.zero_grad()

        ## loss of discriminator
        d_loss = WD(discriminator(real_imgs), discriminator(gen_imgs))

        d_loss.backward()
        optimizer_D.step()


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.epochs, i, len(trainloader), d_loss.item(), g_loss.item())
            )
        
        batches_done = epoch * len(trainloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
