from typing import Dict


def accuracy_from_eval_results(accuracy: Dict) -> float:
    training_exps = max(accuracy.keys())

    accuracy_sum = 0
    accuracy_count = 0
    for t in range(training_exps + 1):
        for e in range(t + 1):
            accuracy_sum += accuracy[t][e]
            accuracy_count += 1

    return accuracy_sum / accuracy_count


def bwt_from_eval_results(accuracy: Dict):
    training_exps = max(accuracy.keys())

    bwt_sum = 0
    bwt_count = 0

    for t in range(training_exps + 1):
        for e in range(t):
            task_bwt = accuracy[t][e] - accuracy[t - 1][e]
            bwt_sum += task_bwt
            bwt_count += 1

    return bwt_sum / bwt_count


def fwt_from_eval_results(accuracy: Dict):
    training_exps = max(accuracy.keys())

    fwt_sum = 0
    fwt_count = 0

    for t in range(training_exps):
        for e in range(t + 1, training_exps + 1):
            task_fwt = accuracy[t][e]
            fwt_sum += task_fwt
            fwt_count += 1

    return fwt_sum / fwt_count
