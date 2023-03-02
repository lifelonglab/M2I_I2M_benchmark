from avalanche.benchmarks.datasets import MNIST
from avalanche.benchmarks.classic import PermutedMNIST
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize, Resize

from paths import DATA_PATH
from scenarios.utils import transform_from_gray_to_rgb, balance_dataset, load_dataset

train_transform = Compose([
    RandomCrop(28, padding=4),
    ToTensor(),
    transform_from_gray_to_rgb(),
    Normalize((0.1307,), (0.3081,))
])

test_transform = Compose([
    ToTensor(),
    transform_from_gray_to_rgb(),
    Normalize((0.1307,), (0.3081,))
])

train_transform_with_resize = Compose([
    ToTensor(),
    Resize((32, 32)),
    transform_from_gray_to_rgb(),
    Normalize((0.1307,), (0.3081,))
])

test_transform_with_resize = Compose([
    ToTensor(),
    Resize((32, 32)),
    transform_from_gray_to_rgb(),
    Normalize((0.1307,), (0.3081,))
])


def load_mnist(balanced=True, number_of_samples_per_class=None):
    if not balanced:
        train = MNIST(root=f'{DATA_PATH}/data', download=True, train=True, transform=train_transform)
        test = MNIST(root=f'{DATA_PATH}/data', download=True, train=False, transform=test_transform)

        return train, test

    train = MNIST(root=f'{DATA_PATH}/data', download=True, train=True)
    test = MNIST(root=f'{DATA_PATH}/data', download=True, train=False)

    train = balance_dataset(train, train_transform, number_of_samples_per_class)
    test = balance_dataset(test, test_transform, number_of_samples_per_class)

    return train, test


def load_mnist_with_resize(balanced=False, number_of_samples_per_class=None):
    train = load_dataset(
        lambda transform: MNIST(root=f'{DATA_PATH}/data', download=True, train=True, transform=transform),
        balanced=balanced, number_of_samples_per_class=number_of_samples_per_class,
        transform=train_transform_with_resize)

    test = load_dataset(
        lambda transform: MNIST(root=f'{DATA_PATH}/data', download=True, train=False, transform=transform),
        balanced=balanced, number_of_samples_per_class=number_of_samples_per_class,
        transform=test_transform_with_resize)

    return train, test
