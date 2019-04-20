import torchvision.datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import dataset

def get_data_loader(dataset_location, batch_size):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader