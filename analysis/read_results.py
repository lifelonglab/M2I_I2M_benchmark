import pathlib
from typing import Optional

import pathlib
from typing import Optional

import pandas as pd
import yaml

from analysis.metrics.eval_results_metrics import accuracy_from_eval_results, \
    bwt_from_eval_results, fwt_from_eval_results, final_accuracy_from_eval_results, mean_accuracy_from_eval_results
from analysis.utils import process_eval_results
from paths import LOGS_PATH, ROOT_PATH

results_path = f'{LOGS_PATH}'


def load_and_save_results():
    df = read_results()
    out_dir = pathlib.Path(f'{ROOT_PATH}/out/')
    out_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_dir / 'results.csv')


def read_results() -> pd.DataFrame:
    data = []
    for path_string in pathlib.Path(results_path).rglob('*'):
        path = pathlib.Path(path_string)
        print(path)
        if path.is_dir() and (path / 'config.yml').exists():
            single_dir_results = _load_single_dir_results(path)
            data.append(single_dir_results)

    return pd.DataFrame(data)


def _load_yaml(path: pathlib.Path):
    with open(path) as f:
        f.readline()  # skip first line that contains some python namespace info
        return yaml.safe_load(f)


def _load_eval_results(path: pathlib.Path) -> Optional[pd.DataFrame]:
    if path.exists():
        df = pd.read_csv(path)
        if len(df) == 0 or max(df['training_exp']) != max(df['eval_exp']):
            return None
        return df
    else:
        print(f'Cannot read eval results: there is no {path} file')
        return None


def _load_single_dir_results(path: pathlib.Path):
    config = _load_yaml(path / 'config.yml')
    balanced = config['balanced'] if 'balanced' in config else 'not_balanced'
    number_of_samples_per_class = config[
        'number_of_samples_per_class'] if 'number_of_samples_per_class' in config else 'all'
    eval_results_metrics = _process_eval_results_to_metrics(path)

    base_info = {'scenario': f"{config['scenario']}_{balanced}_{number_of_samples_per_class}",
                 'model': config['model_name'],
                 'scenario_type': config['scenario_type'], 'strategy': config['strategy_name'], 'source': str(path)}

    return {**base_info, **eval_results_metrics}


def _process_eval_results_to_metrics(path: pathlib.Path):
    eval_results_df = _load_eval_results(path / 'eval_results.csv')
    if eval_results_df is None or len(eval_results_df) == 0:
        return {
            'eval_results_accuracy': -1,
            'eval_results_bwt': -1,
            'eval_results_fwt': -1,
            'eval_results_final_accuracy': -1,
            'eval_results_mean_accuracy': -1
        }

    eval_results = process_eval_results(eval_results_df)
    return {
        'eval_results_accuracy': accuracy_from_eval_results(eval_results),
        'eval_results_bwt': bwt_from_eval_results(eval_results),
        'eval_results_fwt': fwt_from_eval_results(eval_results),
        'eval_results_final_accuracy': final_accuracy_from_eval_results(eval_results),
        'eval_results_mean_accuracy': mean_accuracy_from_eval_results(eval_results),
    }


if __name__ == '__main__':
    load_and_save_results()
