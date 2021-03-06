import torch.utils.data as data_utils
from torchvision.datasets import utils
import os
import numpy as np

import matplotlib.pyplot as plt

MNIST_IMAGE_SIZE = 28

def binarized_mnist_data_loader(dataset_location, batch_size):
    URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    splitdata = []
    for splitname in ["train", "valid", "test"]:
        filename = "binarized_mnist_%s.amat" % splitname
        filepath = os.path.join(dataset_location, filename)
        utils.download_url(URL + filename, dataset_location)
        with open(filepath) as f:
            lines = f.readlines()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
        dataset_loader = data_utils.DataLoader(x, batch_size=batch_size, shuffle=splitname == "train")
        splitdata.append(dataset_loader)
    return splitdata


if __name__ == '__main__':
    train, valid, test = binarized_mnist_data_loader("binarized_mnist", 64)
    for x in train:
        plt.imshow(x[0, 0])
        break
    plt.show()
