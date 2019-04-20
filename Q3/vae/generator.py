import argparse
import os

import torch.utils.data
from torchvision.utils import save_image

from Q3.vae.model import VAE

parser = argparse.ArgumentParser(description='VAE SVHN Image generator')
parser.add_argument('--num-samples', type=int, default=128, metavar='N',
                    help='number of samples to generate (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--saved-model', type=str, default="params_epoch_24_loss_86.3193.pt", metavar='N',
                    help='saved VAE model to generate samples')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


current_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = '{}/results'.format(current_dir)
saved_model = os.path.join(current_dir, 'saved_model', args.saved_model)
print("Loading model {}".format(saved_model))



model = VAE(100).to(device)
model.load_state_dict(torch.load(saved_model, map_location=device), strict=False)
model.eval()

with torch.no_grad():
    sample = torch.randn(args.num_samples, 100).to(device)
    sample = model.generate(sample).cpu()
    print('saving to ', '{}/sample_{}.png'.format(results_dir, args.num_samples))
    save_image(sample.view(args.num_samples, 3, 32, 32),
               '{}/sample_{}.png'.format(results_dir, args.num_samples))


