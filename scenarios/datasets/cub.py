from avalanche.benchmarks.datasets import CUB200

from torchvision.transforms import Resize, Compose, ToTensor, RandomCrop, Normalize

from paths import DATA_PATH

train_transform = Compose([
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


def load_cube200():
    train = CUB200(root=f'{DATA_PATH}/data', download=True, train=True, transform=train_transform)
    test = CUB200(root=f'{DATA_PATH}/data', download=True, train=False, transform=test_transform)

    return train, test


def load_resized_cube200():
    train = CUB200(root=f'{DATA_PATH}/data', download=True, train=True, transform=train_transform_with_resize)
    test = CUB200(root=f'{DATA_PATH}/data', download=True, train=False, transform=test_transform_with_resize)

    return train, test
