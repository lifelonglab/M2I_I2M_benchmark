from avalanche.benchmarks.datasets import CIFAR10
from torchvision.transforms import Resize, Compose, ToTensor, RandomCrop, Normalize

from paths import DATA_PATH
from scenarios.utils import load_dataset

train_transform = Compose([
    RandomCrop(32, padding=4),
    ToTensor(),
    Normalize(mean=(0.9221,), std=(0.2681,))
])

test_transform = Compose([
    ToTensor(),
    Normalize(mean=(0.9221,), std=(0.2681,))
])

train_transform_with_resize = Compose([
    ToTensor(),
    Resize((64, 64)),
    Normalize(mean=(0.9221,), std=(0.2681,))
])

test_transform_with_resize = Compose([
    ToTensor(),
    Resize((64, 64)),
    Normalize(mean=(0.9221,), std=(0.2681,))
])


def load_cifar10():
    train = CIFAR10(root=f'{DATA_PATH}/data', download=True, train=True, transform=train_transform)
    test = CIFAR10(root=f'{DATA_PATH}/data', download=True, train=False, transform=test_transform)

    return train, test


def load_resized_cifar10(balanced=False, number_of_samples_per_class=None):
    train = load_dataset(
        lambda transform: CIFAR10(root=f'{DATA_PATH}/data', download=True, train=True, transform=transform),
        train_transform_with_resize, balanced=balanced,
        number_of_samples_per_class=number_of_samples_per_class)
    test = load_dataset(
        lambda transform: CIFAR10(root=f'{DATA_PATH}/data', download=True, train=False, transform=transform),
        test_transform_with_resize, balanced=balanced,
        number_of_samples_per_class=number_of_samples_per_class)
    return train, test
