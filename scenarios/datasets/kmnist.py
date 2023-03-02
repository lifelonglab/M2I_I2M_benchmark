from avalanche.benchmarks.datasets import KMNIST
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize, transforms, Resize

from paths import DATA_PATH
from scenarios.utils import transform_from_gray_to_rgb, load_dataset

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


def load_kmnist():
    train = KMNIST(root=f'{DATA_PATH}/data', download=True, train=True, transform=train_transform)
    test = KMNIST(root=f'{DATA_PATH}/data', download=True, train=False, transform=test_transform)

    return train, test


def load_kmnist_with_resize(balanced=False, number_of_samples_per_class=None):
    train = load_dataset(
        lambda transform: KMNIST(root=f'{DATA_PATH}/data', download=True, train=True, transform=transform),
        transform=train_transform_with_resize, balanced=balanced,
        number_of_samples_per_class=number_of_samples_per_class)
    test = load_dataset(
        lambda transform: KMNIST(root=f'{DATA_PATH}/data', download=True, train=False, transform=transform),
        transform=test_transform_with_resize, balanced=balanced,
        number_of_samples_per_class=number_of_samples_per_class)

    return train, test
