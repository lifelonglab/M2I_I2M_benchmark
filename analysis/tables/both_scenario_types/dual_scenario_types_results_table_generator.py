import pathlib
from typing import Dict, List

import pandas as pd

from analysis.tables.translations import EVAL_METRICS_TRANSLATIONS
from analysis.utils import select_results, make_ranks
from paths import ROOT_PATH


class DualScenarioTypesResultsGenerator:
    def __init__(self, scenario, strategies, model, class_incremental_metrics, task_incremental_metrics,
                 multiple_results_resolution):
        self.scenario = scenario
        self.strategies = strategies
        self.model = model
        self.ci_metrics = class_incremental_metrics
        self.ti_metrics = task_incremental_metrics
        self.multiple_results_resolution = multiple_results_resolution

    def generate_table(self):
        out_dir = pathlib.Path(f'{ROOT_PATH}/out/')

        df = pd.read_csv(out_dir / 'results.csv')
        class_incremental_results = select_results(df, scenario_type='class_incremental', model=self.model,
                                                   multiple_results_resolution=self.multiple_results_resolution)
        task_incremental_results = select_results(df, scenario_type='task_incremental', model=self.model,
                                                  multiple_results_resolution=self.multiple_results_resolution)

        lines = []
        lines += self._generate_headers()
        lines += self._generate_result_rows(class_incremental_results, task_incremental_results)
        lines += self._generate_bottom()

        tables_out_dir = out_dir / 'tables-dual-scenarios'
        tables_out_dir.mkdir(parents=True, exist_ok=True)

        self._save_to_file(lines, tables_out_dir)

    def _save_to_file(self, lines: List[str], out_dir: pathlib.Path):
        with open(
                out_dir / f'results_{self.scenario}_{self.model}.tex',
                'w') as f:
            for line in lines:
                f.write(line + '\n')

    def _generate_headers(self):
        columns_no = len(self.ci_metrics) + len(self.ti_metrics)
        return [
            # "\\scalebox{.9}{\\sc",
            f"\\begin{{tabular}}{{l{'r' * columns_no}}}",
            "\\toprule",
            " & " + " & ".join(
                [f'\multicolumn{{{len(m)}}}{{c}}{{{s}}}' for s, m in
                 [('Class-incremental', self.ci_metrics), ('Task-incremental', self.ti_metrics)]]) + " \\\\",
            # " ".join(
            #     [f"\\cmidrule(lr){{{str((2 + i * len(self.metrics)))}-{str((1 + (i + 1) * len(self.metrics)))}}}" for
            #      m in range(2)]),
            f"\\cmidrule(lr){{{str(2)}-{str(2 + len(self.ci_metrics) - 1)}}} " + f"\\cmidrule(lr){{{str(2 + len(self.ci_metrics))}-{str(2 - 1 + len(self.ci_metrics) + len(self.ti_metrics))}}} ",
            " & " + " & ".join(
                [EVAL_METRICS_TRANSLATIONS.get(m, m) for m in (self.ci_metrics + self.ti_metrics)]) + "\\\\",
            f"\\midrule"]

    def _generate_bottom(self):
        return [
            "\\bottomrule",
            "\\end{tabular}"
            # "}"  # end scalebox
        ]

    def _generate_row(self, strategy, values: List[str]) -> str:
        return strategy + " & " + " & ".join(values) + " \\\\"

    def _generate_result_rows(self, class_incremental_results: Dict, task_incremental_results: Dict):
        rank_metric = 'eval_results_accuracy'
        class_incremental_ranks = make_ranks(class_incremental_results[self.scenario], strategies=self.strategies,
                                             metric=rank_metric)
        task_incremental_ranks = make_ranks(task_incremental_results[self.scenario], strategies=self.strategies,
                                            metric=rank_metric)

        rows = []

        for strategy in self.strategies:
            strategy_values = []
            for metrics, results, ranks in zip([self.ci_metrics, self.ti_metrics],
                                               [class_incremental_results, task_incremental_results],
                                               [class_incremental_ranks, task_incremental_ranks]):
                for metric in metrics:
                    if metric in results[self.scenario][strategy]:
                        val = results[self.scenario][strategy][metric]
                        if metric == rank_metric:
                            val_str = f"{results[self.scenario][strategy][metric]:.3f} ({ranks[strategy]})" if val > -1 else '?'
                        else:
                            val_str = f"{results[self.scenario][strategy][metric]:.3f}" if val > -1 else '?'
                        strategy_values.append(val_str)
                    else:
                        strategy_values.append('?')
            rows.append(self._generate_row(strategy, strategy_values))

        return rows
