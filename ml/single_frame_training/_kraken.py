from typing import *

from copy import deepcopy



class _KrakenCompatibilityWrap:
    def __init__(self, task, splits: List):
        self.splits = splits
        self.task = task

    def __call__(self, iteration, hyperparameters):
        task = deepcopy(self.task)
        split = self.splits[iteration]
        task.apply_hyperparams(hyperparameters)
        return task._iteration_on_dfs(iteration, split)


def _make_kraken_task(base_task, configs: List[Any], data) -> Tuple[Callable, List[Any]]:
    result = []
    splits = []
    for config in configs:
        task = deepcopy(base_task)
        config = deepcopy(config)
        task.apply_hyperparams(config)
        dfs = task._get_splits(data)
        for d in dfs:
            splits.append(d)
            result.append(dict(hyperparameters=config))
    return _KrakenCompatibilityWrap(deepcopy(base_task), splits), result
