import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
import os


def split(trainset: list, p: float = .8) -> tuple:
    """
    Splits the dataset into training and validation sets.

    Parameters:
    dataset (list): The dataset to be split.
    p (float): Proportion of the dataset to be used as the training set.

    Returns:
    tuple: A tuple containing the training and validation sets.
    """
    train, val = train_test_split(trainset, train_size=p)
    return train, val


def download_celeba(data_dir):
    """Download the CelebA dataset if not already available."""
    datasets.CelebA(root=data_dir, split='all', download=True)
    print("Downloaded CelebA dataset.")


def download_stl10(data_dir):
    """Download the STL10 dataset if not already available."""
    datasets.STL10(root=data_dir, split='all', download=True)
    print("Downloaded STL10 dataset.")


def celeba_subset(dataset: Dataset, feature_idx: int, num_samples: int = -1) -> list:
    """
    Creates a subset of the CelebA dataset based on a specific feature.

    Parameters:
    dataset (Dataset): The CelebA dataset.
    feature_idx (int): The index of the feature to be used for subsetting.
    num_samples (int): The number of samples to include in the subset.

    Returns:
    list: A list of tuples containing the data and corresponding labels.
    """
    NUM_CLASSES = 2
    labelset = {}
    for i in range(NUM_CLASSES):
        one_hot = torch.zeros(NUM_CLASSES)
        one_hot[i] = 1
        labelset[i] = one_hot

    by_class = {}
    features = []
    for idx in tqdm(range(len(dataset))):
        ex, label = dataset[idx]
        features.append(label[feature_idx])
        g = label[feature_idx].numpy().item()
        # ex = torch.mean(ex, dim=0)
        ex = ex.flatten()
        ex = ex / norm(ex)
        if g in by_class:
            by_class[g].append((ex, labelset[g]))
        else:
            by_class[g] = [(ex, labelset[g])]
        if idx > num_samples:
            break
    data = []
    if 1 in by_class:
        max_len = min(25000, len(by_class[1]))
        data.extend(by_class[1][:max_len])
        data.extend(by_class[0][:max_len])
    else:
        max_len = 1
        data.extend(by_class[0][:max_len])
    return data


def get_celeba(feature_idx: int, data_dir: str, batch_size: int, num_train: int = float('inf'), num_test: int = float('inf')) -> tuple:
    """
    Prepares the CelebA dataset for a specific feature.

    Parameters:
    feature_idx (int): The index of the feature to be used.
    split_percentage (float): The percentage of the dataset to be used for training.
    num_train (int): The number of training samples to use.
    num_test (int): The number of test samples to use.

    Returns:
    tuple: A tuple containing DataLoaders for the training, validation, and test sets.
    """
    celeba_path = data_dir
    SIZE = 96
    transform = transforms.Compose(
        [transforms.Resize([SIZE, SIZE]),
         transforms.ToTensor()
         ])

    trainset = torchvision.datasets.CelebA(root=celeba_path,
                                           split='train',
                                           transform=transform,
                                           download=False)
    trainset = celeba_subset(trainset, feature_idx, num_samples=num_train)
    trainset, valset = split(trainset, p=0.2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    testset = torchvision.datasets.CelebA(root=celeba_path,
                                          split='test',
                                          transform=transform,
                                          download=False)
    testset = celeba_subset(testset, feature_idx, num_samples=num_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    print("Train Size: ", len(trainset), "Val Size: ",
          len(valset), "Test Size: ", len(testset))
    return trainloader, valloader, testloader


def download_mnist(data_dir):
    """Download the MNIST dataset if not already available."""
    datasets.MNIST(root=data_dir, train=True, download=True)
    datasets.MNIST(root=data_dir, train=False, download=True)
    print("Downloaded MNIST dataset.")


def get_mnist(data_dir, batch_size):
    """Prepare and load the MNIST dataset, downloading if necessary."""
    mnist_dir = os.path.join(data_dir, 'MNIST')
    if not os.path.exists(mnist_dir):
        download_mnist(data_dir)
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transformer)
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transformer)

    trainset, valset = split(train_dataset, p=0.2)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=1)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_stl_star(data_dir: str, batch_size: int = 128, split_percentage: float = .8, num_train: int = float('inf'), num_test: int = float('inf')) -> tuple:
    """
    Prepares the STL dataset with star shapes added for a binary classification task.

    Parameters:
    split_percentage (float): The percentage of the dataset to be used for training.
    num_train (int): The number of training samples to use.
    num_test (int): The number of test samples to use.

    Returns:
    tuple: A tuple containing DataLoaders for the training, validation, and test sets.
    """
    stl10_dir = os.path.join(data_dir, 'stl10_binary')
    if not os.path.exists(stl10_dir):
        download_stl10(data_dir)
    SIZE = 96
    transform = transforms.Compose(
        [transforms.Resize([SIZE, SIZE]),
         transforms.ToTensor()
         ])

    path = os.path.join(os.getcwd(), 'datasets/')
    trainset = datasets.STL10(root=path,
                              split='train',
                              # train=True,
                              transform=transform,
                              download=True)
    trainset = one_hot_stl_toy(trainset, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)

    valloader = DataLoader(valset, batch_size=batch_size,
                           shuffle=False, num_workers=1)
    testset = datasets.STL10(root=path,
                             split='test',
                             transform=transform,
                             download=True)
    testset = one_hot_stl_toy(testset, num_samples=num_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
          "Num Test: ", len(testset))
    return trainloader, valloader, testloader


def one_hot_stl_toy(dataset: Dataset, num_samples: int = -1) -> list:
    """
    Prepares a toy STL dataset with one-hot encoding.

    Parameters:
    dataset (Dataset): The STL dataset.
    num_samples (int): The number of samples to process.

    Returns:
    list: A list of tuples containing the data and corresponding one-hot encoded labels.
    """
    labelset = {}
    for i in range(2):
        one_hot = torch.zeros(2)
        one_hot[i] = 1
        labelset[i] = one_hot

    subset = [(ex, label) for idx, (ex, label) in enumerate(dataset)
              if idx < num_samples and (label == 0 or label == 9)]

    adjusted = []
    for idx, (ex, label) in enumerate(subset):
        if label == 9:
            ex = draw_star(ex, 1, c=2)
            y = 1
        else:
            ex = draw_star(ex, 0)
            y = 0
        ex = ex.flatten()
        adjusted.append((ex, labelset[y]))
    return adjusted


def draw_star(ex: torch.Tensor, v: float, c: int = 3) -> torch.Tensor:
    """
    Draws a star shape on an image tensor.

    Parameters:
    ex (torch.Tensor): The image tensor.
    v (float): The value to use for drawing the star.
    c (int): The number of channels to draw on.

    Returns:
    torch.Tensor: The image tensor with a star drawn on it.
    """
    ex[:c, 5:6, 7:14] = v
    ex[:c, 4, 9:12] = v
    ex[:c, 3, 10] = v
    ex[:c, 6, 8:13] = v
    ex[:c, 7, 9:12] = v
    ex[:c, 8, 8:13] = v
    ex[:c, 9, 8:10] = v
    ex[:c, 9, 11:13] = v
    return ex
