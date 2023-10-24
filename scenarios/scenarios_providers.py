from dataclasses import dataclass
from typing import Sequence, Union

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import SupportedDataset

from scenarios.datasets.cifar import load_resized_cifar10
from scenarios.datasets.fashion_mnist import load_fashion_mnist_with_resize
from scenarios.datasets.load_imagenet import load_imagenet
from scenarios.datasets.mnist import load_mnist, load_mnist_with_resize
from scenarios.datasets.omniglot import load_resized_omniglot
from scenarios.datasets.svhn import load_svhn_resized
from scenarios.utils import separate_into_tasks, filter_classes


@dataclass
class ScenarioTypeParams:
    task_labels: bool
    class_ids_from_zero_in_each_exp: bool
    class_ids_from_zero_from_first_exp: bool


def _get_params_for(class_incremental: bool) -> ScenarioTypeParams:
    if class_incremental:
        return ScenarioTypeParams(task_labels=False,
                                  class_ids_from_zero_in_each_exp=False,
                                  class_ids_from_zero_from_first_exp=True)
    else:
        return ScenarioTypeParams(task_labels=True,
                                  class_ids_from_zero_in_each_exp=True,
                                  class_ids_from_zero_from_first_exp=False)


def create_scenario(train_dataset: Union[Sequence[SupportedDataset], SupportedDataset],
                    test_dataset: Union[Sequence[SupportedDataset], SupportedDataset],
                    class_incremental: bool):
    params = _get_params_for(class_incremental)

    return nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        one_dataset_per_exp=True,
        task_labels=params.task_labels,
        class_ids_from_zero_in_each_exp=params.class_ids_from_zero_in_each_exp,
        class_ids_from_zero_from_first_exp=params.class_ids_from_zero_from_first_exp,
        shuffle=False,
        n_experiences=0)


def _load_datasets(num_class_from_imagenet=200, num_class_from_emnist=26):
    train_mnist, test_mnist = load_mnist()
    train_emnist_datasets, test_emnist_datasets = separate_into_tasks(*load_emnist(), 10,
                                                                      list(range(1, num_class_from_emnist + 1)))
    train_fashion, test_fashion = load_fashion_mnist_with_resize()
    train_kmnist, test_kmnist = load_kmnist()
    train_imagenet_datasets, test_imagenet_datasets = separate_into_tasks(*load_imagenet(), 10,
                                                                          list(range(num_class_from_imagenet)))
    return test_emnist_datasets, test_fashion, test_imagenet_datasets, test_kmnist, test_mnist, train_emnist_datasets, \
        train_fashion, train_imagenet_datasets, train_kmnist, train_mnist


def get_short_mnist_omniglot_fmnist_svhn_cifar10_imagenet(class_incremental: bool, balanced: bool,
                                                          number_of_samples_per_class=None):
    train_mnist, test_mnist = load_mnist_with_resize(balanced, number_of_samples_per_class)
    train_omniglot, test_omniglot = filter_classes(*load_resized_omniglot(),
                                                   classes=list(range(10)))
    train_fashion, test_fashion = load_fashion_mnist_with_resize(balanced, number_of_samples_per_class)
    train_svhn, test_svhn = load_svhn_resized(balanced, number_of_samples_per_class)
    train_cifar, test_cifar = load_resized_cifar10(balanced, number_of_samples_per_class)
    train_imagenet, test_imagenet = filter_classes(*load_imagenet(balanced, number_of_samples_per_class),
                                                   classes=list(range(10)))

    return create_scenario(
        train_dataset=[train_mnist, train_omniglot, train_fashion, train_svhn, train_cifar, train_imagenet],
        test_dataset=[test_mnist, test_omniglot, test_fashion, test_svhn, test_cifar, test_imagenet],
        class_incremental=class_incremental)


def get_short_imagenet_cifar10_svhn_fmnist_omniglot_mnist(class_incremental: bool, balanced: bool,
                                                          number_of_samples_per_class=None):
    train_mnist, test_mnist = load_mnist_with_resize(balanced, number_of_samples_per_class)
    train_omniglot, test_omniglot = filter_classes(*load_resized_omniglot(),
                                                   classes=list(range(10)))
    train_fashion, test_fashion = load_fashion_mnist_with_resize(balanced, number_of_samples_per_class)
    train_svhn, test_svhn = load_svhn_resized(balanced, number_of_samples_per_class)
    train_cifar, test_cifar = load_resized_cifar10(balanced, number_of_samples_per_class)
    train_imagenet, test_imagenet = filter_classes(*load_imagenet(balanced, number_of_samples_per_class),
                                                   classes=list(range(10)))

    return create_scenario(
        train_dataset=[train_imagenet, train_cifar, train_svhn, train_fashion, train_omniglot, train_mnist],
        test_dataset=[test_imagenet, test_cifar, test_svhn, test_fashion, test_omniglot, test_mnist],
        class_incremental=class_incremental)


def parse_scenario(args):
    class_incremental = args.scenario_type == 'class_incremental'
    resized = args.resized == 'resized'
    balanced = args.balanced == 'balanced'
    number_of_samples_per_class = args.number_of_samples_per_class

    if args.scenario == 'short_mnist_omniglot_fmnist_svhn_cifar10_imagenet':
        return get_short_mnist_omniglot_fmnist_svhn_cifar10_imagenet(class_incremental, balanced=balanced,
                                                                     number_of_samples_per_class=number_of_samples_per_class)
    elif args.scenario == 'short_imagenet_cifar10_svhn_fmnist_omniglot_mnist':
        return get_short_imagenet_cifar10_svhn_fmnist_omniglot_mnist(class_incremental, balanced=balanced,
                                                                     number_of_samples_per_class=number_of_samples_per_class)
    else:
        raise NotImplementedError("SCENARIO NOT IMPLEMENTED YET: ", args.scenario)


if __name__ == '__main__':
    scenario = get_short_imagenet_cifar10_svhn_fmnist_omniglot_mnist(class_incremental=False, balanced=False)
    for t in scenario.train_stream:
        print(t)
