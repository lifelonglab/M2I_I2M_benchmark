from avalanche.benchmarks.datasets import Omniglot
from avalanche.benchmarks.utils import AvalancheTensorDataset, make_tensor_classification_dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Resize, Compose, ToTensor, RandomCrop, Normalize

from paths import DATA_PATH
from scenarios.utils import transform_from_gray_to_rgb, load_dataset

transform_not_resize = Compose([
    RandomCrop(105, padding=4),
    ToTensor(),
    transform_from_gray_to_rgb(),
    Normalize(mean=(0.9221,), std=(0.2681,))
])

transform_with_resize = Compose([
    ToTensor(),
    Resize((64, 64)),
    transform_from_gray_to_rgb(),
    Normalize(mean=(0.9221,), std=(0.2681,))
])


def _load_omniglot(transform_func, balanced: bool = False, number_of_samples_per_class=None):
    dataset = Omniglot(root=f'{DATA_PATH}/data', download=True, train=True)
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.targets, train_size=0.8, test_size=0.2)
    train = load_dataset(lambda transform: AvalancheTensorDataset(X_train, y_train, transform=transform),
                         transform=transform_func, balanced=balanced,
                         number_of_samples_per_class=number_of_samples_per_class)
    test = load_dataset(lambda transform: AvalancheTensorDataset(X_test, y_test, transform=transform),
                        transform=transform_func, balanced=balanced,
                        number_of_samples_per_class=number_of_samples_per_class)
    return train, test


def load_omniglot():
    return _load_omniglot(transform_not_resize)


def load_resized_omniglot():
    return _load_omniglot(transform_with_resize)


if __name__ == '__main__':
    _load_omniglot(transform_not_resize)
