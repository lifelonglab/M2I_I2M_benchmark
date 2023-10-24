import math
from collections import defaultdict
from typing import List, Tuple, Callable, Optional

from avalanche.benchmarks.utils import AvalancheDataset, AvalancheTensorDataset
from avalanche.benchmarks.utils.dataset_utils import ClassificationSubset
from torchvision.transforms import transforms


def _filter_classes_in_single_dataset(dataset, classes):
    indices = [i for i, t in enumerate(dataset.targets) if t in classes]
    max_class = max(classes)
    class_mapping = [-1] * (max_class + 1)
    for i, c in enumerate(sorted(classes)):
        class_mapping[c] = i

    return AvalancheDataset(ClassificationSubset(dataset, indices, class_mapping=class_mapping))


def filter_classes(train_dataset, test_dataset, classes):
    return _filter_classes_in_single_dataset(train_dataset, classes), _filter_classes_in_single_dataset(test_dataset,
                                                                                                        classes)


def _split_classes_list(classes, no_classes_in_task) -> List:
    return [classes[i * no_classes_in_task: (i + 1) * no_classes_in_task] for i in
            range(math.floor(len(classes) / no_classes_in_task))]


def separate_into_tasks(train_dataset, test_dataset, no_classes_in_task, targets_set) -> Tuple[List, List]:
    split = _split_classes_list(targets_set, no_classes_in_task)
    tasks = [filter_classes(train_dataset, test_dataset, classes_in_task) for classes_in_task in split]
    x = [train for train, _ in tasks], [test for _, test in tasks]
    return x


def transform_from_gray_to_rgb():
    return transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)


def balance_dataset(dataset, transform, number_of_samples_per_class=None):
    img_occurrences = defaultdict(lambda: 0)
    X = []
    y = []

    for data in dataset:
        target = data[1]
        if img_occurrences[target] < number_of_samples_per_class:
            X.append(data[0])
            y.append(target)
            img_occurrences[target] += 1

        if all(i == number_of_samples_per_class for i in img_occurrences): break

    return AvalancheTensorDataset(X, y, transform=transform)


def load_dataset(dataset_loader: Callable[[Optional[Callable]], AvalancheDataset], transform: Optional[Callable],
                 balanced: bool, number_of_samples_per_class=None):
    if balanced:
        return balance_dataset(dataset_loader(None), transform, number_of_samples_per_class)
    else:
        return dataset_loader(transform)
