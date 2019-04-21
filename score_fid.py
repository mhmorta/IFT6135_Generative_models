import argparse
import os
from scipy import linalg
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]

def mean_covariance(extracted_feature):
    mean = extracted_feature.mean(dim=1)
    covariance = extracted_feature.var(dim=1)
    return mean, covariance

def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    """
    To be implemented by you!
    """
    mu_p = mu_q = []



    for x in sample_feature_iterator:
        mu1, sigma1 = mean_covariance(x)

    for x in testset_feature_iterator:
        mu2, sigma2 = mean_covariance(x)






    # assert mu1.shape == mu2.shape
    # assert sigma1.shape == sigma2.shape

    # diff = mu1 - mu2

    # covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp = False)
    # if not np.isfinite(covmean).all():
    #     msg = ('fid calculation produces singular product')
    #     print (msg)
    #     offset = np.eye(sigma1.shape[0]) * 1e-6
    #     covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # if np.iscomplexobj(covmean):
    #     if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
    #         m = np.max(np.abs(covmean.imag))
    #         raise ValueError('Imaginary component {}'.format(m))
    #     covmean = covmean.real

    # trace_covmean = np.trace(covmean)

    # d2 = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - (2 * trace_covmean))

    raise NotImplementedError(
        "TO BE IMPLEMENTED."
        "Part of Assignment 3 Quantitative Evaluations"
    )
    return d2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
