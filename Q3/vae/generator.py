import argparse
import os

import torch.utils.data
from torchvision.utils import save_image

from Q3.vae.model import VAE

parser = argparse.ArgumentParser(description='VAE SVHN Image generator')
parser.add_argument('--num-samples', type=int, default=1000, metavar='N',
                    help='number of samples to generate (default: 128)')
parser.add_argument('--sample-merged', type=bool, default=False, metavar='N',
                    help='Whether we want one big sample')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--saved-model', type=str, default="params_epoch_29_loss_81.2645.pt", metavar='N',
                    help='saved VAE model to generate samples')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


current_dir = os.path.dirname(os.path.realpath(__file__))
saved_model = os.path.join(current_dir, 'best_model', args.saved_model)
print("Loading model {}".format(saved_model))


model = VAE(100).to(device)
model.load_state_dict(torch.load(saved_model, map_location=device), strict=False)
model.eval()

with torch.no_grad():
    sample = torch.randn(args.num_samples, 100).to(device)
    sample = model.generate(sample).cpu()
    images = sample.view(args.num_samples, 3, 32, 32)
    if args.sample_merged:
        save_image(images, '{}/sample_merged.png'
                   .format('{}/sample_merged/'.format(current_dir)), nrow=10, normalize=True)
    else:
        for idx, img in enumerate(images):
            file_name = '{}/sample_{}.png'.format('{}/samples/data'.format(current_dir), idx)
            print(idx, 'saving to', file_name)
            save_image(img, file_name, normalize=True)

