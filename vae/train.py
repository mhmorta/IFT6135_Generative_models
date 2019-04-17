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
parser.add_argument('--max-batch-idx', type=int, default=99999, metavar='N',
                    help='only for debugging locally')
parser.add_argument('--hidden-features', type=int, default=100, metavar='N',
                    help='latent variable size')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
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

optimizer = optim.Adam(model.parameters(), lr=3e-4)

current_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = '{}/results'.format(current_dir)
saved_model = '{}/saved_model'.format(current_dir)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # E[log P(X|z)]
    # fidelity loss
    # https://youtu.be/Hnns75GNUzs?list=PLdxQ7SoCLQANizknbIiHzL_hYjEaI-wUe&t=608
    # todo reduce_sum or reduce_mean? https://youtu.be/Hnns75GNUzs?list=PLdxQ7SoCLQANizknbIiHzL_hYjEaI-wUe&t=739
    logx_z_likelihood = -F.binary_cross_entropy(recon_x.view(args.batch_size, -1), x.view(args.batch_size, -1), reduction='sum')

    # Compute the divergence D_KL[q(z|x)||p(z)]
    # given z ~ N(0, 1)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # todo if we change it to negative value adjust the saving best model logic in main

    ELBO = logx_z_likelihood - KLD

    # optimizer will minimize loss function, thus in order to maximize ELBO we have to negate it, i.e loss = -ELBO
    return -ELBO


def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        if batch_idx > args.max_batch_idx:
            break
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
    print('====> Average Validation loss: {:.4f}'.format(test_loss))
    return test_loss


def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Average Test loss: {:.4f}'.format(test_loss))


def sample(epoch):
    model.eval()
    with torch.no_grad():
        sample = torch.randn(args.batch_size, args.hidden_features).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(args.batch_size, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE),
                   '{}/sample_{}.png'.format(results_dir, epoch))


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = binarized_mnist_data_loader('{}/binarized_mnist'.format(current_dir),
                                                                          args.batch_size)
    best_valid_loss = None
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        new_valid_loss = validate(epoch)
        if best_valid_loss is None or new_valid_loss < best_valid_loss:
            best_valid_loss = new_valid_loss
            print('Saving model with avg loss {}'.format(best_valid_loss))
            torch.save(model.state_dict(),
                       os.path.join(saved_model, 'params_epoch_{}_loss_{:.4f}.pt'.format(epoch, best_valid_loss)))

        sample(epoch)

    test()
