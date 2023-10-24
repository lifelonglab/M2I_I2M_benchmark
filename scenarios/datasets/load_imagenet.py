from avalanche.benchmarks.datasets import TinyImagenet
from torchvision.transforms import Resize, Compose, ToTensor

from paths import DATA_PATH
from scenarios.utils import load_dataset


def load_resized_imagenet():
    train = TinyImagenet(root=f'{DATA_PATH}/data', train=True, transform=Compose([ToTensor(), Resize((64, 64))]))
    test = TinyImagenet(root=f'{DATA_PATH}/data', train=False, transform=Compose([ToTensor(), Resize((64, 64))]))

    return train, test


def load_imagenet(balanced=False, number_of_samples_per_class=None):
    train = load_dataset(lambda transform: TinyImagenet(root=f'{DATA_PATH}/data', train=True, transform=transform),
                         transform=Compose([ToTensor()]), balanced=balanced,
                         number_of_samples_per_class=number_of_samples_per_class)
    test = load_dataset(lambda transform: TinyImagenet(root=f'{DATA_PATH}/data', train=False, transform=transform),
                        transform=Compose([ToTensor()]), balanced=balanced,
                        number_of_samples_per_class=number_of_samples_per_class)

    return train, test
