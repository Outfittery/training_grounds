from yo_fluq_ds import FileIO
from pathlib import Path

from .architecture import FileCacheTrainingEnvironment
from ..packaging import EntryPoint



def _myprint(s):
    print(s, flush=True)

def execute(entry: EntryPoint):
    folder = Path('/opt/ml/model')
    env = FileCacheTrainingEnvironment(_myprint, folder)

    hyperparams = FileIO.read_json('/opt/ml/input/config/hyperparameters.json')
    if '_tuning_objective_metric' in hyperparams:
        del hyperparams['_tuning_objective_metric']

    model = env.common_initialization(entry, Path('/opt/ml/code/package.tar.gz'),{}, 'model')
    env.log('Starting training now...')
    model.run_with_environment('/opt/ml/input/data/training/', env)

