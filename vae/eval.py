import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm

from vae.train import model, device
from vae.dataloader import MNIST_IMAGE_SIZE
from torch.nn import functional as F



# based on
# http://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/
# https://github.com/bjlkeng/sandbox/blob/master/notebooks/vae-importance_sampling/vae-mnist-importance-sampling.ipynb
def sample_z(data, num_samples):
    mus, logvars = model.encode(data)
    mus = mus.detach().numpy()
    logvars = logvars.detach().numpy()

    z_samples = []
    qz = []

    for mu, logvar in zip(mus, logvars):
        z_vals = [model.reparameterize(torch.from_numpy(mu), torch.from_numpy(logvar)).detach().numpy() for _ in range(num_samples)]
        z_samples.append(z_vals)
        qz.append([norm.pdf(zv, loc=mu, scale=np.exp(0.5 * logvar)) for zv in z_vals])

    z_samples = np.array(z_samples)
    qz = np.array(qz)
    return z_samples, qz


def eval_log_px(x, num_samples):
    z_samples, qz = sample_z(x, num_samples)
    pz = norm.pdf(z_samples)
    z_samples = torch.from_numpy(z_samples).float()
    result = []
    for x_input, z_sample, pz_i, qz_i in zip(x, z_samples, pz, qz):
        x_recon = model.decode(z_sample)
        logp_xz = []
        for x_r in x_recon:
            logp_xz.append(-F.binary_cross_entropy(x_r, x_input, reduction='sum'))
        logp_xz = np.array(logp_xz)
        # See: scipy.special.logsumexp
        # \log p(x) = E_p[p(x|z)]
        # = \log(\int p(x|z) p(z) dz)
        # = \log(\int p(x|z) p(z) / q(z|x) q(z|x) dz)
        # = E_q[p(x|z) p(z) / q(z|x)]
        # ~= \log(1/n * \sum_i p(x|z_i) p(z_i)/q(z_i))
        # = \log p(x) = \log(1/n * \sum_i e^{\log p(x|z_i) + \log p(z_i) - \log q(z_i)})
        # = \log p(x) = -\logn + \logsumexp_i(\log p(x|z_i) + \log p(z_i) - \log q(z_i))
        log_pz = np.sum(np.log(pz_i), axis=-1)
        log_qz = np.sum(np.log(qz_i), axis=-1)
        argsum = logp_xz + log_pz - log_qz
        log_px = -np.log(len(argsum)) + logsumexp(argsum)
        result.append(log_px)

    return np.array(result)


with torch.no_grad():
    # load examples from the valid_loader to save time because it takes too long to load
    model.load_state_dict(torch.load('best_model/params_epoch_20_loss_94.4314.pt', map_location=device))
    model.eval()
    for i in range(10):
        print('Batch ', i)
        X = torch.load('sample_input/batch_size_64/valid_{:03d}.pt'.format(i))
        ret = np.mean(eval_log_px(X, num_samples=200))
        print('log(x): ', ret)
        elbo = -model.loss_function(X, *model(X))
        print('ELBO:', elbo.item())
