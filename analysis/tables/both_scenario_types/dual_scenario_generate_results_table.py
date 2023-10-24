from dataclasses import dataclass, field
from typing import List

from analysis.read_results import load_and_save_results
from analysis.tables.both_scenario_types.dual_scenario_types_results_table_generator import \
    DualScenarioTypesResultsGenerator

DEFAULT_CI_METRICS = ['eval_results_accuracy', 'eval_results_bwt']
DEFAULT_TI_METRICS = ['eval_results_accuracy', 'eval_results_bwt', 'eval_results_fwt']
# , 'eval_results_final_accuracy', 'eval_results_mean_accuracy']
DEFAULT_SCENARIOS = ['short_mnist_omniglot_fmnist_svhn_cifar10_imagenet',
                     'short_imagenet_cifar10_svhn_fmnist_omniglot_mnist']
DEFAULT_STRATEGIES = ['AGEM', 'Cumulative', 'CWRStar', 'EWC', 'GenerativeReplay', 'GEM', 'GDumb', 'LwF', 'MAS', 'Naive',
                      'Replay', 'SI']

DEFAULT_STRATEGIES_ORDERED = ['EWC', 'LwF', 'MAS', 'SI', 'AGEM', 'GEM', 'GenerativeReplay', 'Replay', 'CWRStar', 'PNN',
                              'Naive', 'GDumb', 'Cumulative']
DEFAULT_MULTIPLE_RESULTS_RESOLUTION = 'max_accuracy'  # max_accuracy | error

scenarios_not_balanced = ['short_mnist_omniglot_fmnist_svhn_cifar10_imagenet_not_balanced_all',
                          'short_imagenet_cifar10_svhn_fmnist_omniglot_mnist_not_balanced_all']

scenarios_500 = ['short_mnist_omniglot_fmnist_svhn_cifar10_imagenet_balanced_500',
                 'short_imagenet_cifar10_svhn_fmnist_omniglot_mnist_balanced_500']

scenarios_5000 = ['short_mnist_omniglot_fmnist_svhn_cifar10_imagenet_balanced_5000',
                  'short_imagenet_cifar10_svhn_fmnist_omniglot_mnist_balanced_5000']


@dataclass
class ResultsTableConfig:
    model: str
    scenario: str
    multiple_results_resolution: str = DEFAULT_MULTIPLE_RESULTS_RESOLUTION  # max_accuracy | error
    strategies: List[str] = field(default_factory=lambda: DEFAULT_STRATEGIES_ORDERED)
    ci_metrics: List[str] = field(default_factory=lambda: DEFAULT_CI_METRICS)
    ti_metrics: List[str] = field(default_factory=lambda: DEFAULT_TI_METRICS)


TABLES_TO_GENERATE = [
    # ResultsTableConfig(model='EfficientNet', scenarios=scenarios_500),
    # ResultsTableConfig(model='EfficientNet_NotPretrained', scenarios=scenarios_500),
    # ResultsTableConfig(model='EfficientNet_NotPretrained', scenarios=scenarios_5000),
    # ResultsTableConfig(model='EfficientNet', scenarios=scenarios_500),
    # ResultsTableConfig(model='EfficientNet_NotPretrained', scenarios=scenarios_500),
    # ResultsTableConfig(model='EfficientNet_NotPretrained', scenarios=scenarios_5000),
    # ResultsTableConfig(model='EfficientNet', scenarios=scenarios_500),
    # ResultsTableConfig(model='EfficientNet_NotPretrained', scenarios=scenarios_500),
    # ResultsTableConfig(model='EfficientNet_multihead', scenarios=scenarios_500),
    # ResultsTableConfig(model='EfficientNet_MultiHead_NotPretrained',
    #                    scenario=scenarios_500),

    ResultsTableConfig(model='wide_VGG9', scenario=scenarios_500[0]),
    ResultsTableConfig(model='wide_VGG9', scenario=scenarios_500[1]),
    ResultsTableConfig(model='EfficientNet_NotPretrained', scenario=scenarios_500[0]),
    ResultsTableConfig(model='EfficientNet_NotPretrained', scenario=scenarios_500[1]),
    ResultsTableConfig(model='ResNet', scenario=scenarios_500[0]),
    ResultsTableConfig(model='ResNet', scenario=scenarios_500[1]),

    # ResultsTableConfig(model='wide_VGG99_multihead', scenario_type='task_incremental'),
]

if __name__ == '__main__':
    load_and_save_results()
    for table in TABLES_TO_GENERATE:
        g = DualScenarioTypesResultsGenerator(scenario=table.scenario, strategies=table.strategies, model=table.model,
                                              class_incremental_metrics=table.ci_metrics,
                                              task_incremental_metrics=table.ti_metrics,
                                              multiple_results_resolution=table.multiple_results_resolution)
        g.generate_table()
