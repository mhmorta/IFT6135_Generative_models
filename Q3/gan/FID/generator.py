import argparse
import os

import torch.utils.data
from torchvision.utils import save_image

from models_org import Generator

from Q3.gan.FID.models_org import sample_generator

parser = argparse.ArgumentParser(description='VAE SVHN Image generator')
parser.add_argument('--num-samples', type=int, default=1000, metavar='N',
                    help='number of samples to generate (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--saved-model', type=str, default="gan_svhn_model.pt", metavar='N',
                    help='saved VAE model to generate samples')

print(torch.__version__)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
cuda = torch.cuda.is_available()


current_dir = os.path.dirname(os.path.realpath(__file__))
samples_dir = '{}/samples/data'.format(current_dir)
saved_model = os.path.join(current_dir, args.saved_model)
print("Loading model {}".format(saved_model))


model = Generator(3, 100, cuda).to(device)
model.load_state_dict(torch.load(saved_model, map_location=device), strict=False)
model.eval()

with torch.no_grad():
    num_samples = 1000
    sample = sample_generator(model, num_samples, 100, device).cpu()
    for idx, img in enumerate(sample.view(num_samples, 3, 32, 32)):
        file_name = '{}/sample_{}.png'.format(samples_dir, idx)
        print(idx, 'saving to', file_name)
        save_image(img, file_name, normalize=True)


