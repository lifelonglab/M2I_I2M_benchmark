import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.read_results import _load_yaml

sns.set_theme()
sns.set(rc={'figure.figsize': (8, 6)})


def plot_heatmap(results_filepath, output_file_path):
    df = pd.read_csv(results_filepath)
    if len(df) == 0:
        return
    experiences_no = max(df['training_exp']) + 1
    df = df.pivot(index='training_exp', columns='eval_exp', values='eval_accuracy')

    masks = np.zeros((experiences_no, experiences_no))
    for i in range(experiences_no):
        for j in range(i + 1, experiences_no):
            masks[i, j] = True

    p = sns.heatmap(df, annot=True, vmin=0, vmax=1, center=0.5, mask=masks,
                    cmap=sns.color_palette('plasma', as_cmap=True))
    p.set_xlabel('Evaluating on experience')
    p.set_ylabel('After learning experience')

    config = _load_yaml(pathlib.Path(results_filepath).parent / 'config.yml')
    plt.title(f"{config['strategy_name']} - {config['scenario']} - {config['scenario_type']}")
    plt.savefig(f'{output_file_path}.pdf', bbox_inches='tight')
    # plt.show()

    plt.close()


if __name__ == '__main__':
    for path_string in pathlib.Path('logs/').rglob('*'):
        path = pathlib.Path(path_string)
        if path.is_dir() and (path / 'eval_results.csv').exists():
            try:
                plot_heatmap(path / 'eval_results.csv', path)
            except Exception:
                print(f'Could not plot for {path}')
