from vae.train import current_dir, model, device
from vae.dataloader import binarized_mnist_data_loader, MNIST_IMAGE_SIZE
import numpy as np
from scipy.stats import norm
import torch


# based on
# http://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/
# https://github.com/bjlkeng/sandbox/blob/master/notebooks/vae-importance_sampling/vae-mnist-importance-sampling.ipynb
def compute_samples(data, num_samples, debug=False):
    """ Sample from importance distribution z_samples ~ q(z|X) and
        compute p(z_samples), q(z_samples) for importance sampling
    """
    model.eval()
    recon_batch, z_mean, z_log_sigma = model(data)

    z_samples = []
    qz = []

    for m, s in zip(z_mean, z_log_sigma):
        m = m.detach().numpy()
        s = s.detach().numpy()
        z_vals = [np.random.normal(m[i], np.exp(s[i]), num_samples)
                  for i in range(len(m))]
        qz_vals = [norm.pdf(z_vals[i], loc=m[i], scale=np.exp(s[i]))
                   for i in range(len(m))]
        z_samples.append(z_vals)
        qz.append(qz_vals)

    z_samples = np.array(z_samples)
    pz = norm.pdf(z_samples)
    qz = np.array(qz)

    z_samples = np.swapaxes(z_samples, 1, 2)
    pz = np.swapaxes(pz, 1, 2)
    qz = np.swapaxes(qz, 1, 2)

    if debug:
        print(z_mean.shape, z_log_sigma.shape)
        print('m, s', m[0], s[0])
        print('samples', z_samples[-1][0])
        print('pvals', pz[-1][0])
        print('qvals', qz[-1][0])

        print(z_samples.shape)
        print(pz.shape)
        print(qz.shape)

    return z_samples, pz, qz


# i put in here 10 examples from the valid_loader to save time because it takes too long to load
X = torch.load('sample_input/X.pt')
compute_samples(X, 4, debug=True)