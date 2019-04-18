from vae.dataloader import binarized_mnist_data_loader
from vae.train import current_dir, args
import torch

_, valid_loader, _ = binarized_mnist_data_loader('{}/binarized_mnist'.format(current_dir), args.batch_size)

for batch_idx, data in enumerate(valid_loader):
    torch.save(data, '{}/sample_input/batch_size_{}/valid_{:03d}.pt'.
               format(current_dir, args.batch_size, batch_idx))
