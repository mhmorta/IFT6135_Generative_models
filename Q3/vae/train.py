import argparse
import os

import torch
import torch.utils.data
from torch import optim
from torchvision.utils import save_image

from Q3.vae.dataloader import svhn_data_loader
from Q3.vae.model import VAE

parser = argparse.ArgumentParser(description='VAE SVHN Example')
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


def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, y) in enumerate(train_loader):
        if batch_idx > args.max_batch_idx:
            break
        data = data.to(device)
        optimizer.zero_grad()
        _, mean_x, mu, logvar = model(data)
        loss = model.loss_function(data, mean_x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    train_loss /= (batch_idx + 1)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss))


def validate(epoch):
    model.eval()
    valid_loss = 0

    with torch.no_grad():
        for i, (data, y) in enumerate(valid_loader):
            data = data.to(device)
            z, mean_x, mu, logvar = model(data)
            valid_loss += model.loss_function(data, mean_x, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], model.generate(z)[:n]])
                save_image(comparison.cpu(),
                         '{}/reconstruction_{}.png'.format(results_dir, epoch), nrow=n)

    valid_loss /= (i + 1)
    print('====> Average Validation loss: {:.4f}'.format(valid_loss))
    return valid_loss


def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            z, mean_x, mu, logvar = model(data)
            test_loss += model.loss_function(data, mean_x, mu, logvar).item()

    test_loss /= (i + 1)
    print('====> Average Test loss: {:.4f}'.format(test_loss))


def sample(epoch):
    model.eval()
    with torch.no_grad():
        sample = torch.randn(args.batch_size, args.hidden_features).to(device)
        sample = model.generate(sample).cpu()
        save_image(sample.view(args.batch_size, 3, 32, 32),
                   '{}/sample_{}.png'.format(results_dir, epoch))

if __name__ == '__main__':

    train_loader, valid_loader, test_loader = svhn_data_loader("svhn", args.batch_size)
    best_valid_loss = None
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        new_valid_loss = validate(epoch)
        if best_valid_loss is None or new_valid_loss < best_valid_loss:
            best_valid_loss = new_valid_loss
            print('Saving model with avg loss {}'.format(best_valid_loss))
            torch.save(model.state_dict(),
                       os.path.join(saved_model, 'params_epoch_{}_loss_{:.4f}.pt'.format(epoch, best_valid_loss)))

        #sample(epoch)

    test()
