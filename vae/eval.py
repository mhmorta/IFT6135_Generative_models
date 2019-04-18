import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm
from torch.nn import functional as F
import os

from vae.train import model, device, current_dir


# based on
# http://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/
# https://github.com/bjlkeng/sandbox/blob/master/notebooks/vae-importance_sampling/vae-mnist-importance-sampling.ipynb
def sample_z(model, X, num_samples):
    mus, logvars = model.encode(X)
    z_samples = []
    qz = []

    for mu, logvar in zip(mus, logvars):
        z_vals = torch.stack([model.reparameterize(mu, logvar) for _ in range(num_samples)])
        z_samples.append(z_vals)
        qz.append([norm.pdf(zv.cpu(), loc=mu.cpu(), scale=np.exp(0.5 * logvar.cpu())) for zv in z_vals])
    z_samples = torch.stack(z_samples)
    qz = np.array(qz)
    return z_samples, qz


def eval_log_px(model, X, z_samples, qz):
    pz = norm.pdf(z_samples.cpu())
    result = []
    for x_input, z_sample, pz_i, qz_i in zip(X, z_samples, pz, qz):
        x_recon = model.decode(z_sample)
        log_pxz = []
        for x_r in x_recon:
            log_pxz.append(-F.binary_cross_entropy(x_r, x_input, reduction='sum'))
        log_pxz = np.array(log_pxz)
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
        argsum = log_pxz + log_pz - log_qz
        log_px = -np.log(len(argsum)) + logsumexp(argsum)
        result.append(log_px)

    return np.array(result)


with torch.no_grad():
    # load examples from the valid_loader to save time because it takes too long to load
    print('loading trained model')
    model.load_state_dict(torch.load('{}/best_model/params_epoch_20_loss_94.4314.pt'.format(current_dir), map_location=device))
    model.eval()
    log_px_arr = []
    elbo_arr = []
    dir_name = '{}/split_mnist/batch_size_64/'.format(current_dir)
    for file in sorted(os.listdir(os.fsencode(dir_name))):
        filename = os.fsdecode(file)
        print('Batch for', filename)
        X = torch.load('{}/{}'.format(dir_name, filename), map_location=device)
        z_samples, qz = sample_z(model, X, num_samples=200)
        ret = np.mean(eval_log_px(model, X, z_samples, qz))
        print('log p(x): ', ret)
        log_px_arr.append(ret)
        elbo = -model.loss_function(X, *model(X)).item()
        print('ELBO:', elbo)
        elbo_arr.append(elbo)
    print('===FINAL===')
    print('log p(x)={}, ELBO={}'.format(np.mean(log_px_arr), np.mean(elbo_arr)))

