"""
Code based on the vae example at: https://raw.githubusercontent.com/pytorch/examples/master/vae/main.py
"""
import argparse
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from vae.model import VAE
from vae.dataloader import binarized_mnist_data_loader, MNIST_IMAGE_SIZE
from torch.nn import functional as F
import os


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--hidden-features', type=int, default=100, metavar='N',
                    help='latent variable size')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

model = VAE(args.hidden_features).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

current_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = '{}/results'.format(current_dir)

train_loader, valid_loader, test_loader = binarized_mnist_data_loader('{}/binarized_mnist'.format(current_dir), args.batch_size)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # TODO confirm if we do need this?
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #return BCE + KLD

    return BCE

def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))



def validate(epoch):
    model.eval()
    test_loss = 0


    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)[:n]])
                save_image(comparison.cpu(),
                         '{}/reconstruction_{}.png'.format(results_dir, epoch), nrow=n)

    test_loss /= len(valid_loader.dataset)
    print('====> Validation set loss: {:.4f}'.format(test_loss))


def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()


    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def sample(epoch):
    with torch.no_grad():
        sample = torch.randn(args.batch_size, args.hidden_features).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(args.batch_size, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE),
                   '{}/sample_{}.png'.format(results_dir, epoch))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validate(epoch)
    sample(epoch)

test()