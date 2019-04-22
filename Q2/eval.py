import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm
from torch.nn import functional as F

from Q2.dataloader import binarized_mnist_data_loader
from Q2.train import model, device, current_dir


# based on
# http://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/
# https://github.com/bjlkeng/sandbox/blob/master/notebooks/vae-importance_sampling/vae-mnist-importance-sampling.ipynb
def sample_z(model, x, num_samples):
    mus, logvars = model.encode(x)
    z_samples = []
    qz = []
    for mu, logvar in zip(mus, logvars):
        z_vals = torch.stack([model.reparameterize(mu, logvar) for _ in range(num_samples)])  # (K x L)
        # multiplying logvar by 0.5 as in the VAE.loss_function() because \log \sigma^2 = 2 \log \sigma
        qz_vals = [norm.pdf(zv.cpu(), loc=mu.cpu(), scale=np.exp(0.5 * logvar.cpu())) for zv in z_vals]  # (K x L)
        z_samples.append(z_vals)
        qz.append(qz_vals)
    z_samples = torch.stack(z_samples)  # (M x K x L)
    qz = np.array(qz)  # (M x K x L)
    return z_samples, qz


def eval_log_px(model, x, z_samples, qz):
    pz = norm.pdf(z_samples.cpu())  # (M x K x L)
    ret = []
    # input: (1 x 28 x 28)
    for x_input, z_sample, pz_i, qz_i in zip(x, z_samples, pz, qz):
        x_recon = model.decode(z_sample)  # x_recon: (K x 1 x 28 x 28)
        log_pxz = []
        for x_r in x_recon:
            log_pxz.append(-F.binary_cross_entropy(x_r, x_input, reduction='sum').cpu())
        log_pxz = np.array(log_pxz)  # (K)
        log_pz = np.sum(np.log(pz_i), axis=-1)  # (K)
        log_qz = np.sum(np.log(qz_i), axis=-1)  # (K)
        argsum = log_pxz + log_pz - log_qz  # (K)
        log_px = -np.log(len(argsum)) + logsumexp(argsum)
        ret.append(log_px)

    ret = np.array(ret)  # (M)
    return ret


if __name__ == '__main__':
    with torch.no_grad():
        print('loading trained model')
        model.load_state_dict(torch.load('{}/best_model/params_epoch_20_loss_94.4314.pt'.format(current_dir), map_location=device))
        model.eval()
        log_px_arr = []
        elbo_arr = []
        _, valid_loader, test_loader = binarized_mnist_data_loader('{}/binarized_mnist'.format(current_dir), 64)

        for loader in [('validation', valid_loader), ('test', test_loader)]:
            print('Running on dataset:', loader[0])
            for batch_idx, data in enumerate(loader[1]):
                data = data.to(device)
                z_samples, qz = sample_z(model, data, num_samples=200)
                ret = np.mean(eval_log_px(model, data, z_samples, qz))
                log_px_arr.append(ret)
                elbo = -model.loss_function(data, *model(data)).item()
                elbo_arr.append(elbo)
                if batch_idx % 10 == 0:
                    print('Batch id', batch_idx)
                    print('ELBO:', elbo)
                    print('log p(x): ', ret)
            print('===FINAL===', loader[0])
            print('log p(x)={}, ELBO={}'.format(np.mean(log_px_arr), np.mean(elbo_arr)))

