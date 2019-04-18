import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm

from vae.train import model, device


# based on
# http://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/
# https://github.com/bjlkeng/sandbox/blob/master/notebooks/vae-importance_sampling/vae-mnist-importance-sampling.ipynb
def sample_z(data, num_samples, debug=False):
    """ Sample from importance distribution z_samples ~ q(z|X) and
        compute p(z_samples), q(z_samples) for importance sampling
    """
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
    pz = norm.pdf(z_samples)
    qz = np.array(qz)

    if debug:
        print(mus.shape, logvars.shape)
        print('m, s', mu[0], logvar[0])
        print('samples', z_samples[-1][0])
        print('pvals', pz[-1][0])
        print('qvals', qz[-1][0])

        print(z_samples.shape)
        print(pz.shape)
        print(qz.shape)

    return z_samples, pz, qz


def eval_log_px(x, num_samples, debug=False):
    z_samples, pz, qz = sample_z(x, num_samples)
    z_samples = torch.from_numpy(z_samples).float()
    assert len(z_samples) == len(x)
    assert len(z_samples) == len(pz)
    assert len(z_samples) == len(qz)

    # Calculate importance sample
    # \log p(x) = E_p[p(x|z)]
    # = \log(\int p(x|z) p(z) dz)
    # = \log(\int p(x|z) p(z) / q(z|x) q(z|x) dz)
    # = E_q[p(x|z) p(z) / q(z|x)]
    # ~= \log(1/n * \sum_i p(x|z_i) p(z_i)/q(z_i))
    # = \log p(x) = \log(1/n * \sum_i e^{\log p(x|z_i) + \log p(z_i) - \log q(z_i)})
    # = \log p(x) = -\logn + \logsumexp_i(\log p(x|z_i) + \log p(z_i) - \log q(z_i))
    # See: scipy.special.logsumexp
    result = []
    for i in range(len(x)):
        datum = x[i].reshape(784).detach().numpy()
        x_predict = model.decode(z_samples[i]).reshape(-1, 784).detach().numpy()
        x_predict = np.clip(x_predict, np.finfo(float).eps, 1. - np.finfo(float).eps)
        p_vals = pz[i]
        q_vals = qz[i]

        # \log p(x|z) = Binary cross entropy
        logp_xz = np.sum(datum * np.log(x_predict) + (1. - datum) * np.log(1.0 - x_predict), axis=-1)
        logpz = np.sum(np.log(p_vals), axis=-1)
        logqz = np.sum(np.log(q_vals), axis=-1)
        argsum = logp_xz + logpz - logqz
        logpx = -np.log(num_samples) + logsumexp(argsum)
        result.append(logpx)

        if debug:
            print(x_predict.shape)
            print(p_vals.shape)
            print(q_vals.shape)
            print(logp_xz.shape)
            print(logpz.shape)
            print(logqz.shape)
            print("logp_xz", logp_xz)
            print("logpz", logpz)
            print("logqz", logqz)
            print(argsum.shape)
            print("logpx", logpx)

    return np.array(result)


with torch.no_grad():
    # load examples from the valid_loader to save time because it takes too long to load
    model.load_state_dict(torch.load('best_model/params_epoch_20_loss_94.4314.pt', map_location=device))
    model.eval()
    for i in range(10):
        print('Batch ', i)
        X = torch.load('sample_input/batch_size_64/valid_{:03d}.pt'.format(i))
        ret = np.mean(eval_log_px(X, num_samples=128, debug=False))
        print('log(x): ', ret)
        elbo = -model.loss_function(X, *model(X))
        print('ELBO:', elbo.item())
