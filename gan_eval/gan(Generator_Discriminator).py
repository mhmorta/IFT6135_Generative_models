import argparse
import os
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
from torchvision.utils import save_image
import samplers as samplers



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


class Generator(nn.Module):
    def __init__(self, channels, latent_dim, cuda):
        super(Generator, self).__init__()
        self.cuda = cuda

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            )

        self.convTranspose = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.25),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.25),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.5),
            nn.ReLU(),

            nn.ConvTranspose2d(8, channels, 3, 1, 1),
            nn.Tanh()
            )
        self.init_weights()
    def init_weights(self):
        for m in self.convTranspose:
            if isinstance(m,nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if type(self.mlp) == nn.Linear:
            nn.init.xavier_uniform_(self.mlp.weight)
            self.mlp.bias.data.fill_(0.01)


    def forward(self, input):
        x = self.mlp(input)
        x = x.view(-1, 128, 4, 4)
        return self.convTranspose(x)

class Discriminator(nn.Module):
    def __init__(self, channels, latent_dim, cuda):
        super(Discriminator, self).__init__()
        self.cuda = cuda

        self.conv = nn.Sequential( 
            nn.Conv2d(channels, 8, 3, 1, 1),
            # nn.BatchNorm2d(8),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(16, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            # nn.Dropout2d(0.5),
            # nn.LeakyReLU(0.2),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(0.2),            
            )
        self.mlp = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1)
            )

        self.init_weights()
    def init_weights(self):
        for m in self.conv:
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if type(self.mlp) == nn.Linear:
            nn.init.xavier_uniform_(self.mlp.weight)
            self.mlp.bias.data.fill_(0.01)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * 4 * 4)
        return self.mlp(x)


def sample_generator(Generator, num_samples, latent_dim, update_d, device):
    noise = Variable(torch.randn(num_samples, latent_dim), requires_grad=False).to(device)
    noise.require_grad = False

    gen_samples = Generator(noise)
    gen_samples = gen_samples.view(-1, 3, 32, 32)
    save_image(gen_samples.data.view(num_samples, 3, 32, 32).cpu(), 'results/gs' + str(update_d) + '.png', nrow = 8)

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
    # optimizer_G = torch.optim.Adam(Generator.parameters(), lr=opt.lr*4)
    # optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=opt.lr)
    optimizer_G = torch.optim.Adam(Generator.parameters(), lr=opt.lr*4, betas=(0.5, 0.999))
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
    D.to(device)
    G.to(device)

    trainloader, validloader, testloader = get_data_loader('./data', 512)
    train(D, G, trainloader, opt.latent_dim, opt.batch_size, opt.epochs, device)



# ## Generator
# class Generator(nn.Module):
#     """docstring for Generator"""
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.model = nn.Sequntial(
#             nn.Conv2d(3, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.Dropout2d(p=0.25),
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Dropout2d(p=0.25),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Dropout2d(p=0.25),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Dropout2d(p=0.25),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Dropout2d(p=0.25),
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.Dropout2d(p=0.25),
#             nn.MaxPool2d(2),

#             nn.Conv2d(128, 512, 2),
#             )
#         self.mlp = nn.Sequntial(
#             nn.ReLU(),
#             nn.Dropout2d(p=0.5),
#             nn.Linear(512,10),
#             )
#     def forward(self, x):
#         return self.mlp(self.model(x)[:,:,0, 0])

# ## Discriminator
# class Discriminator(nn.Module):
#     """docstring for Discriminator"""
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequntial(
#             nn.Linear(2, opt.latent_dim),
#             nn.ReLU(),
#             nn.Linear(opt.latent_dim, opt.latent_dim),
#             nn.ReLU(),
#             nn.Linear(opt.latent_dim, 1),
#             nn.Sigmoid(),
#             )
#     def forward(self, x):
#         return self.model(x)


# def WD(D_x, D_y, X, Y):
#     lam = opt.lamda
#     D_loss_real = torch.mean(D_x)
#     D_loss_fake = torch.mean(D_y)
#     D_loss = D_loss_real - D_loss_fake - (lam * gradient_penalty)
#     return -D_loss
  
# def gradient_penalty(X,Y):
#     batch_size = X.size()[0]
#     a = np.random.uniform(0, 1)
#     z = a * X + (1-a) * Y
#     Z = Variable(z, require_grad=True)

#     prob_z = Discriminator(Z)

#     # Calculate gradients of probabilities with respect to examples
#     gradients = torch_grad(outputs=prob_z, inputs=z, grad_outputs=torch.ones(prob_z.size()).cuda() if cuda else torch.ones(
#                            prob_z.size()),
#                            create_graph=True, retain_graph=True)[0]
#     gradients = gradients.view(batch_size, -1)
#     gardient_pen = ((gradients.norm(2, dim=1) -1)**2).mean()
#     return gardient_pen


# ## Initializing the generator and discriminator
# generator = Generator()
# discriminator = Discriminator()

# if cuda():
#     generator.cuda()
#     discriminator.cuda()


# ## optimizers
# optimizer_G = torch.optim.opt.optimizer(generator.parameters(), lr=opt.lr)
# optimizer_D = torch.optim.opt.optimizer(discriminator.parameters(), lr=opt.lr)


# ## training
# for epoch in range(opt.epoch):
#     for i,(imgs, _) in enumerate(trainloader):

#         real = Variable(Tensor(imgs.size(0), 1).fill_(1.0), require_grad=False)
#         fake = Variable(Tensor(imgs.size(0), 1).fill_(1.0), require_grad=False)

#         real_imgs = Variable(imgs.type(Tensor))

#         # -----------------
#         ## Train generator
#         # -----------------

#         optimizer_G.zero_grad()

#         noise = Variable(Tensor(samplers.distribution1(0, opt.batch_size)))

#         ## generate a batch of images
#         gen_imgs = generator(noise)

#         ## loss of generator (what is the adversarial loss)
#         g_loss = ????

#         g_loss.backward()
#         optimizer_G.step()


#         # ---------------------
#         ## Train discriminator
#         # ---------------------

#         optimizer_D.zero_grad()

#         ## loss of discriminator
#         d_loss = WD(discriminator(real_imgs), discriminator(gen_imgs))

#         d_loss.backward()
#         optimizer_D.step()


#         print(
#             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#             % (epoch, opt.epochs, i, len(trainloader), d_loss.item(), g_loss.item())
#             )
        
#         batches_done = epoch * len(trainloader) + i
#         if batches_done % opt.sample_interval == 0:
#             save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
