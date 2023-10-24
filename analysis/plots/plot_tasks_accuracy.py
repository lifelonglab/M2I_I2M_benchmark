import pathlib

import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.read_results import load_and_save_results
from analysis.utils import process_eval_results, select_results
from paths import ROOT_PATH

sns.set_theme()
sns.set(rc={'figure.figsize': (8, 6)})

g_model = 'EfficientNet_multihead'
g_scenario = 'mnist_to_imagenet_short'
g_strategies_to_include = ['AGEM', 'CoPE', 'Cumulative', 'CWRStar', 'EWC', 'GEM', 'GDumb', 'LFL', 'LwF', 'MAS',
                           'Replay', 'SI']
g_scenario_type = 'task_incremental'
g_multiple_results_resolution = 'max_accuracy'  # max_accuracy | error


def plot_tasks_accuracy(scenario, strategies, model, scenario_type, multiple_results_resolution):
    out_dir = pathlib.Path(f'{ROOT_PATH}/out/')

    df = pd.read_csv(out_dir / 'results.csv')
    results = select_results(df, scenario_type=scenario_type, model=model,
                             multiple_results_resolution=multiple_results_resolution)

    colors = sns.color_palette() + sns.color_palette("pastel")
    patches = []

    for strategy, color in zip(strategies, colors):
        if 'source' in results[scenario][strategy]:
            strategy_source = results[scenario][strategy]['source']
            _plot_single_method(pathlib.Path(strategy_source), color)
            patches.append(mpatches.Patch(color=color, label=strategy))
        else:
            print(f'No results for {scenario} and {strategy}')

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title(f'{scenario} - {model} - {scenario_type}')

    plot_dir = out_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{plot_dir}/tasks_accuracy_{scenario}_{scenario_type}_{model}.pdf', bbox_inches='tight')
    # plt.show()


def _plot_single_method(results_file: pathlib.Path, color):
    df = pd.read_csv(results_file / 'eval_results.csv')
    results = process_eval_results(df)
    if len(results) > 0:
        for i in range(max(results.keys()) + 1):
            xs = range(1, i + 2)
            ys = []
            for j in range(i + 1):
                ys.append(results[i][j])
            sns.lineplot(x=xs, y=ys, markers=True, marker='o', color=color)


if __name__ == '__main__':
    load_and_save_results()
    plot_tasks_accuracy(scenario=g_scenario, strategies=g_strategies_to_include, model=g_model,
                        scenario_type=g_scenario_type, multiple_results_resolution=g_multiple_results_resolution)
