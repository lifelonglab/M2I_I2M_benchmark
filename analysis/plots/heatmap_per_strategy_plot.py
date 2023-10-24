import pathlib
from collections import defaultdict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.utils import process_eval_results, select_results
from paths import ROOT_PATH

sns.set_theme()
sns.set_style(rc={'figure.figsize': (12, 9)})

g_models = ['wide_VGG9', 'ResNet', 'EfficientNet_NotPretrained']
g_scenarios = [
    'short_mnist_omniglot_fmnist_svhn_cifar10_imagenet_balanced_500',
    'short_imagenet_cifar10_svhn_fmnist_omniglot_mnist_balanced_500',
    # 'short_mnist_omniglot_fmnist_svhn_cifar10_imagenet_balanced_5000',
    # 'short_imagenet_cifar10_svhn_fmnist_omniglot_mnist_balanced_5000',

]
g_strategies_to_include = ['Cumulative', 'Naive', 'AGEM', 'CWRStar', 'EWC', 'GEM', 'GDumb'
                                                                                   'Replay', 'GenerativeReplay', 'LwF',
                           'MAS', 'PNN', 'SI']
g_scenario_types = ['class_incremental', 'task_incremental']
g_multiple_results_resolution = 'max_accuracy'  # max_accuracy | error


def _plot_single_heatmap(data, color, **kwargs):
    new_df = data.copy()
    del new_df['Strategy']
    sns.heatmap(new_df, annot=False, vmin=0, vmax=1, cmap=sns.color_palette('plasma', as_cmap=True), **kwargs)


def plot_tasks_heatmap(scenario, strategies, model, tasks_to_plot, scenario_type, multiple_results_resolution):
    print(f'Plotting for {scenario} and {model} and {scenario_type}')
    sns.set(font_scale=1.5)
    out_dir = pathlib.Path(f'{ROOT_PATH}/out/')

    df = pd.read_csv(out_dir / 'results.csv')
    selected_results = select_results(df, scenario_type=scenario_type, model=model,
                                      multiple_results_resolution=multiple_results_resolution)

    all_results = {}
    for strategy in strategies:
        if 'source' in selected_results[scenario][strategy]:
            strategy_source = pathlib.Path(selected_results[scenario][strategy]['source'])
            df = pd.read_csv(strategy_source / 'eval_results.csv')
            all_results[strategy] = process_eval_results(df)

    subplots_no = len(tasks_to_plot)

    eval_tasks_to_show = None
    dfs = []
    for strategy in strategies:
        strategy_result = defaultdict(dict)
        if strategy in all_results:
            for task_to_plot in tasks_to_plot:
                for i in range(task_to_plot, max(all_results[strategy].keys()) + 1):
                    strategy_result[task_to_plot][i] = all_results[strategy][i][task_to_plot]

        strategy_df = pd.DataFrame.from_dict(strategy_result)
        strategy_df['Strategy'] = strategy
        dfs.append(strategy_df)

    df = pd.concat(dfs)

    g = sns.FacetGrid(df, col="Strategy", col_wrap=6, sharex=True)
    cbar_ax = g.fig.add_axes([.92, .3, .02, .4])  # <-- Create a colorbar axes

    g.map_dataframe(_plot_single_heatmap, cbar_ax=cbar_ax)
    g.fig.subplots_adjust(left=0.07, right=.9)  # <-- Add space so the colorbar doesn't overlap the plot
    g.set_axis_labels(y_var="Training task")
    g.set_titles(col_template="{col_name}")

    plt.savefig(f'{out_dir}/heatmaps/strategies_{model}_{scenario}_{scenario_type}.pdf', bbox_inches='tight', dpi=300)

    # plt.show()


if __name__ == '__main__':
    # load_and_save_results()
    for g_model in g_models:
        for g_scenario in g_scenarios:
            for g_scenario_type in g_scenario_types:
                plot_tasks_heatmap(scenario=g_scenario, strategies=g_strategies_to_include, model=g_model,
                                   tasks_to_plot=[0, 1, 2, 3, 4, 5],
                                   scenario_type=g_scenario_type,
                                   multiple_results_resolution=g_multiple_results_resolution)
