from avalanche.benchmarks.datasets import TinyImagenet
from torchvision.transforms import Resize, Compose, Grayscale, ToTensor

from paths import DATA_PATH
from scenarios.utils import load_dataset


def load_greyscaled_resized_imagenet():
    train = TinyImagenet(root=f'{DATA_PATH}/data', train=True,
                         transform=Compose([ToTensor(), Resize((28, 28)), Grayscale()]))
    test = TinyImagenet(root=f'{DATA_PATH}/data', train=False,
                        transform=Compose([ToTensor(), Resize((28, 28)), Grayscale()]))

    return train, test


def load_greyscaled_imagenet():
    train = TinyImagenet(root=f'{DATA_PATH}/data', train=True,
                         transform=Compose([ToTensor(), Grayscale()]))
    test = TinyImagenet(root=f'{DATA_PATH}/data', train=False,
                        transform=Compose([ToTensor(), Grayscale()]))

    return train, test


def load_resized_imagenet():
    train = TinyImagenet(root=f'{DATA_PATH}/data', train=True, transform=Compose([ToTensor(), Resize((28, 28))]))
    test = TinyImagenet(root=f'{DATA_PATH}/data', train=False, transform=Compose([ToTensor(), Resize((28, 28))]))

    return train, test


def load_imagenet(balanced=False, number_of_samples_per_class=None):
    train = load_dataset(lambda transform: TinyImagenet(root=f'{DATA_PATH}/data', train=True, transform=transform),
                         transform=Compose([ToTensor()]), balanced=balanced,
                         number_of_samples_per_class=number_of_samples_per_class)
    test = load_dataset(lambda transform: TinyImagenet(root=f'{DATA_PATH}/data', train=False, transform=transform),
                        transform=Compose([ToTensor()]), balanced=balanced,
                        number_of_samples_per_class=number_of_samples_per_class)

    return train, test


if __name__ == '__main__':
    train, test = load_imagenet()
    # test, _ = load_greyscaled_resized_imagenet()
    print(set(test.targets))
