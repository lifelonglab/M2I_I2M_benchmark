from avalanche.benchmarks.datasets import SVHN
from torchvision.transforms import ToTensor, Resize, Compose

from paths import DATA_PATH
from scenarios.utils import load_dataset


def _load_svhn(split: str, transform):
    dataset = SVHN(root=f'{DATA_PATH}/data', split=split, download=True, transform=transform)
    dataset.targets = dataset.labels
    return dataset


def load_svhn():
    train = _load_svhn('train', Compose([ToTensor()]))
    test = _load_svhn('test', Compose([ToTensor()]))

    return train, test


def load_svhn_resized(balanced=False, number_of_samples_per_class=None):
    transform_func = Compose([ToTensor(), Resize((64, 64))])
    train = load_dataset(lambda transform: _load_svhn('train', transform), transform_func, balanced=balanced,
                         number_of_samples_per_class=number_of_samples_per_class)
    test = load_dataset(lambda transform: _load_svhn('test', transform), transform_func, balanced=balanced,
                        number_of_samples_per_class=number_of_samples_per_class)

    return train, test
