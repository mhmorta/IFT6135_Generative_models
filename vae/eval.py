from vae.train import current_dir, model, device
from vae.dataloader import binarized_mnist_data_loader, MNIST_IMAGE_SIZE
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
import torch


# based on
# http://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/
# https://github.com/bjlkeng/sandbox/blob/master/notebooks/vae-importance_sampling/vae-mnist-importance-sampling.ipynb
def compute_samples(data, num_samples, debug=False):
    """ Sample from importance distribution z_samples ~ q(z|X) and
        compute p(z_samples), q(z_samples) for importance sampling
    """
    z_mean, z_log_sigma = model.encode(data)

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
#compute_samples(X, 4, debug=True)


def estimate_logpx_batch(data, num_samples, debug=False):
    z_samples, pz, qz = compute_samples(data, num_samples)
    z_samples = torch.from_numpy(z_samples).float()
    assert len(z_samples) == len(data)
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
    for i in range(len(data)):
        datum = data[i].reshape(784).detach().numpy()
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


model.eval()
model.load_state_dict(torch.load('saved_model/params_epoch_9_loss_98.7259.pt', map_location=device))
ret = estimate_logpx_batch(X, num_samples=128, debug=False)
print('log(x): ', ret)
