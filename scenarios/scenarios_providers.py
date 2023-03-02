from dataclasses import dataclass
from typing import Sequence, Union

from avalanche.benchmarks import nc_benchmark, SplitMNIST, NCScenario, SplitTinyImageNet, PermutedMNIST
from avalanche.benchmarks.utils import SupportedDataset
from torchvision.transforms import ToTensor, Compose

from scenarios.datasets.cifar import load_cifar10, load_resized_cifar10
from scenarios.datasets.cub import load_resized_cube200, load_cube200
from scenarios.datasets.emnist_datasets import load_emnist, load_emnist_with_resize
from scenarios.datasets.fashion_mnist import load_fashion_mnist_with_resize
from scenarios.datasets.kmnist import load_kmnist, load_kmnist_with_resize
from scenarios.datasets.load_imagenet import load_imagenet
from scenarios.datasets.mnist import load_mnist, load_mnist_with_resize
from scenarios.datasets.omniglot import load_omniglot, load_resized_omniglot
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



def get_mnist_scenario(class_incremental=True):
    train_mnist, test_mnist = load_mnist_with_resize()
    train_emnist_datasets, test_emnist_datasets = separate_into_tasks(*load_emnist_with_resize(), 10,
                                                                      list(range(1, 27)))
    train_fashion, test_fashion = load_fashion_mnist_with_resize()
    train_kmnist, test_kmnist = load_kmnist_with_resize()

    return create_scenario(train_dataset=[train_mnist, *train_emnist_datasets, train_fashion, train_kmnist],
                           test_dataset=[test_mnist, *test_emnist_datasets, test_fashion, test_kmnist],
                           class_incremental=class_incremental)


def get_simple_emist() -> NCScenario:
    train_emnist_datasets, test_emnist_datasets = separate_into_tasks(*load_emnist(), 10, list(range(1, 27)))

    return nc_benchmark(train_dataset=[*train_emnist_datasets],
                        test_dataset=[*test_emnist_datasets],
                        train_transform=ToTensor(),
                        eval_transform=ToTensor(),
                        one_dataset_per_exp=True,
                        task_labels=False,
                        shuffle=False,
                        class_ids_from_zero_in_each_exp=False,
                        class_ids_from_zero_from_first_exp=True,
                        n_experiences=0)


def get_simple_mnist(class_incremental=True, resized=False, balanced=False, number_of_samples_per_class=None):
    train_mnist_datasets, test_mnist_datasets = load_mnist_with_resize(balanced, number_of_samples_per_class) \
        if resized else load_mnist(balanced, number_of_samples_per_class)

    if class_incremental:
        scenario = nc_benchmark(
            train_mnist_datasets,
            test_mnist_datasets,
            n_experiences=5,
            shuffle=True,
            seed=1234,
            task_labels=True,
            fixed_class_order=list(range(10)),
            class_ids_from_zero_in_each_exp=False,
            class_ids_from_zero_from_first_exp=True)
    else:
        scenario = nc_benchmark(
            train_mnist_datasets,
            test_mnist_datasets,
            n_experiences=5,
            shuffle=True,
            seed=1234,
            task_labels=True,
            fixed_class_order=list(range(10)),
            class_ids_from_zero_in_each_exp=True,
            class_ids_from_zero_from_first_exp=False)

    return scenario


def get_cifar_mnist_kmnist_fashionmist_scenario(class_incremental: bool, balanced: bool,
                                                number_of_samples_per_class=None) -> NCScenario:
    train_cifar, test_cifar = load_resized_cifar10(balanced, number_of_samples_per_class)
    train_mnist, test_mnist = load_mnist_with_resize(balanced, number_of_samples_per_class)
    train_kmnist, test_kmnist = load_kmnist_with_resize(balanced, number_of_samples_per_class)
    train_fashion, test_fashion = load_fashion_mnist_with_resize(balanced, number_of_samples_per_class)

    return create_scenario(
        train_dataset=[train_cifar, train_mnist, train_kmnist, train_fashion],
        test_dataset=[test_cifar, test_mnist, test_kmnist, test_fashion],
        class_incremental=class_incremental)


def get_mnist_to_imagenet_scenario(class_incremental=True, num_class_from_imagenet=200,
                                   num_class_from_emnist=26) -> NCScenario:
    test_emnist_datasets, test_fashion, test_imagenet_datasets, test_kmnist, test_mnist, train_emnist_datasets, \
    train_fashion, train_imagenet_datasets, train_kmnist, train_mnist = _load_datasets(num_class_from_imagenet,
                                                                                       num_class_from_emnist)

    return create_scenario(
        train_dataset=[train_mnist, *train_emnist_datasets, train_fashion, train_kmnist, *train_imagenet_datasets],
        test_dataset=[test_mnist, *test_emnist_datasets, test_fashion, test_kmnist, *test_imagenet_datasets],
        class_incremental=class_incremental)


def get_omniglot_scenario(class_incremental=True, num_classes=10, n_class_in_task=2, resized=True) -> NCScenario:
    func = lambda: load_resized_omniglot() if resized else load_omniglot()
    train_omniglot_datasets, test_omniglot_datasets = separate_into_tasks(*func(),
                                                                          n_class_in_task,
                                                                          list(range(num_classes)))

    return create_scenario(train_dataset=[*train_omniglot_datasets], test_dataset=[*test_omniglot_datasets],
                           class_incremental=class_incremental)


def get_cifar10_scenario(class_incremental=True, resized=True) -> NCScenario:
    train_cifar10_datasets, test_cifar10_datasets = load_resized_cifar10() if resized else load_cifar10()

    if class_incremental:
        return nc_benchmark(
            train_dataset=[train_cifar10_datasets],
            test_dataset=[test_cifar10_datasets],
            # train_transform=ToTensor(), eval_transform=ToTensor(),
            one_dataset_per_exp=True,
            class_ids_from_zero_in_each_exp=False,
            class_ids_from_zero_from_first_exp=True,
            task_labels=False,
            shuffle=False,
            n_experiences=0)
    else:
        return nc_benchmark(
            train_dataset=train_cifar10_datasets,
            test_dataset=test_cifar10_datasets,
            # train_transform=ToTensor(), eval_transform=ToTensor(),
            one_dataset_per_exp=True,
            class_ids_from_zero_in_each_exp=True,
            class_ids_from_zero_from_first_exp=False,
            task_labels=True,
            shuffle=False,
            n_experiences=5)


def get_cube200_scenario(class_incremental=True, num_classes=200, n_class_in_task=2, resized=True) -> NCScenario:
    func = lambda: load_resized_cube200() if resized else load_cube200()
    train_cube200_datasets, test_cube200_datasets = separate_into_tasks(*func(),
                                                                        n_class_in_task,
                                                                        list(range(num_classes)))

    return create_scenario(train_dataset=[*train_cube200_datasets],
                           test_dataset=[*test_cube200_datasets], class_incremental=class_incremental)


def get_imagenet_scenario(class_incremental=True, num_classes=200, n_class_in_task=2) -> NCScenario:
    train_imagenet_datasets, test_imagenet_datasets = separate_into_tasks(*load_imagenet(),
                                                                          n_class_in_task,
                                                                          list(range(num_classes)))

    return create_scenario(train_dataset=[*train_imagenet_datasets], test_dataset=[*test_imagenet_datasets],
                           class_incremental=class_incremental)


def get_imagenet_to_mnist_scenario(class_incremental=True, num_class_from_imagenet=200,
                                   num_class_from_emnist=26) -> NCScenario:
    test_emnist_datasets, test_fashion, test_imagenet_datasets, test_kmnist, test_mnist, train_emnist_datasets, \
    train_fashion, train_imagenet_datasets, train_kmnist, train_mnist = _load_datasets(num_class_from_imagenet,
                                                                                       num_class_from_emnist)

    return create_scenario(
        train_dataset=[*train_imagenet_datasets, train_kmnist, train_fashion, *train_emnist_datasets, train_mnist],
        test_dataset=[*test_imagenet_datasets, test_kmnist, test_fashion, *test_emnist_datasets, test_mnist],
        class_incremental=class_incremental)


def get_split_mnist_scenario() -> NCScenario:
    return SplitMNIST(n_experiences=5, return_task_id=True, fixed_class_order=list(range(10)))


def get_permuted_mnist_scenario(class_incremental=True) -> NCScenario:
    if class_incremental:
        return PermutedMNIST(n_experiences=5, return_task_id=False)

    return PermutedMNIST(n_experiences=5, return_task_id=True)


def get_split_imagenet_scenario(class_incremental=True, num_classes=200, n_experiences=40) -> NCScenario:
    if class_incremental:
        return SplitTinyImageNet(n_experiences=n_experiences, return_task_id=False,
                                 fixed_class_order=list(range(num_classes)),
                                 train_transform=Compose([ToTensor()]),
                                 eval_transform=Compose([ToTensor()])
                                 )

    return SplitTinyImageNet(n_experiences=n_experiences, return_task_id=True,
                             fixed_class_order=list(range(num_classes)),
                             train_transform=Compose([ToTensor()]),
                             eval_transform=Compose([ToTensor()])
                             )


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

    if args.scenario == 'split_mnist':
        return get_split_mnist_scenario()
    elif args.scenario == 'permuted_mnist':
        return get_permuted_mnist_scenario(args.scenario_type == 'class_incremental')
    elif args.scenario == 'simple_mnist':
        return get_simple_mnist(args.scenario_type == 'class_incremental',
                                resized=resized,
                                balanced=balanced,
                                number_of_samples_per_class=args.number_of_samples_per_class)
    elif args.scenario == 'all_mnist':
        return get_mnist_scenario(args.scenario_type == 'class_incremental')
    elif args.scenario == 'mnist_to_imagenet_short':
        return get_mnist_to_imagenet_scenario(args.scenario_type == 'class_incremental', 10, 10)
    elif args.scenario == 'mnist_to_imagenet_long':
        return get_mnist_to_imagenet_scenario(args.scenario_type == 'class_incremental', 200, 26)
    elif args.scenario == 'imagenet_to_mnist_short':
        return get_imagenet_to_mnist_scenario(args.scenario_type == 'class_incremental', 10, 10)
    elif args.scenario == 'imagenet_to_mnist_long':
        return get_imagenet_to_mnist_scenario(args.scenario_type == 'class_incremental', 200, 26)
    elif args.scenario == 'simple_emist':
        return get_simple_emist()
    elif args.scenario == 'imagenet':
        n_class_in_task = int(args.num_classes / args.num_experiences)
        return get_imagenet_scenario(args.scenario_type == 'class_incremental',
                                     num_classes=args.num_classes,
                                     n_class_in_task=n_class_in_task)
    elif args.scenario == 'split_imagenet':
        return get_split_imagenet_scenario(args.scenario_type == 'class_incremental',
                                           num_classes=args.num_classes,
                                           n_experiences=args.num_experiences)

    elif args.scenario == 'omniglot':
        n_class_in_task = int(args.num_classes / args.num_experiences)
        return get_omniglot_scenario(args.scenario_type == 'class_incremental',
                                     resized=args.resized == 'resized',
                                     num_classes=args.num_classes,
                                     n_class_in_task=n_class_in_task)
    elif args.scenario == 'cifar10':
        return get_cifar10_scenario(args.scenario_type == 'class_incremental', resized=args.resized == 'resized')
    elif args.scenario == 'cube200':
        n_class_in_task = int(args.num_classes / args.num_experiences)
        return get_cube200_scenario(args.scenario_type == 'class_incremental',
                                    num_classes=args.num_classes,
                                    n_class_in_task=n_class_in_task,
                                    resized=args.resized == 'resized')
    elif args.scenario == 'short_mnist_omniglot_fmnist_svhn_cifar10_imagenet':
        return get_short_mnist_omniglot_fmnist_svhn_cifar10_imagenet(class_incremental, balanced=balanced,
                                                                     number_of_samples_per_class=number_of_samples_per_class)
    elif args.scenario == 'short_imagenet_cifar10_svhn_fmnist_omniglot_mnist':
        return get_short_imagenet_cifar10_svhn_fmnist_omniglot_mnist(class_incremental, balanced=balanced,
                                                                     number_of_samples_per_class=number_of_samples_per_class)
    elif args.scenario == 'cifar_mnist_kmnist_fashionmist':
        return get_cifar_mnist_kmnist_fashionmist_scenario(class_incremental, balanced=balanced,
                                                           number_of_samples_per_class=number_of_samples_per_class)
    else:
        raise NotImplementedError("SCENARIO NOT IMPLEMENTED YET: ", args.scenario)


if __name__ == '__main__':
    # scenario = get_imagenet_scenario(class_incremental=True, num_classes=200, n_class_in_task=40)
    # scenario = get_mnist_to_imagenet_scenario()
    scenario = get_cifar_mnist_kmnist_fashionmist_scenario(class_incremental=False, balanced=False)
    # scenario = get_cifar10_scenario(class_incremental=False)
    # scenario = get_short_imagenet_cifar10_svhn_fmnist_omniglot_mnist(class_incremental=False, balanced=False)
    for t in scenario.train_stream:
        print(t)
