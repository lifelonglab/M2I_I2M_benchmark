import torch
from avalanche.benchmarks.datasets import EMNIST
from torch.utils.data import Subset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize, transforms, Resize

from paths import DATA_PATH
from scenarios.utils import _filter_classes_in_single_dataset, transform_from_gray_to_rgb

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
    Resize((64, 64)),
    transform_from_gray_to_rgb(),
    Normalize((0.1307,), (0.3081,))
])

test_transform_with_resize = Compose([
    ToTensor(),
    Resize((64, 64)),
    transform_from_gray_to_rgb(),
    Normalize((0.1307,), (0.3081,))
])



def load_emnist():
    emnist_train = EMNIST(root=f'{DATA_PATH}/data', split='letters', download=True, train=True,
                          transform=train_transform)
    emnist_test = EMNIST(root=f'{DATA_PATH}/data', split='letters', download=True, train=False,
                         transform=test_transform)
    return emnist_train, emnist_test


def load_emnist_with_resize():
    emnist_train = EMNIST(root=f'{DATA_PATH}/data', split='letters', download=True, train=True,
                          transform=train_transform_with_resize)
    emnist_test = EMNIST(root=f'{DATA_PATH}/data', split='letters', download=True, train=False,
                         transform=test_transform_with_resize)
    return emnist_train, emnist_test


def load_emnist_tasks():
    emnist_train = EMNIST(root=f'{DATA_PATH}/data', split='letters', download=True, train=True)
    emnist_test = EMNIST(root=f'{DATA_PATH}/data', split='letters', download=True, train=False)

    classes_per_task = [list(range(1, 9)), list(range(10, 18)), list(range(20, 27))]
    train_tasks = [_filter_classes_in_single_dataset(emnist_train, classes) for classes in classes_per_task]
    test_tasks = [_filter_classes_in_single_dataset(emnist_test, classes) for classes in classes_per_task]

    return train_tasks, test_tasks
