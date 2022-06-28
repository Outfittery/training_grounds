from yo_fluq_ds import FileIO
from pathlib import Path
import traceback

from .architecture import FileCacheTrainingEnvironment
from ..packaging import EntryPoint
from ..._common.logger import  Logger, SagemakerLoggerInterface



Logger.reset(SagemakerLoggerInterface())

def execute(entry: EntryPoint):
    folder = Path('/opt/ml/model')
    env = FileCacheTrainingEnvironment(folder)

    hyperparams = FileIO.read_json('/opt/ml/input/config/hyperparameters.json')
    if '_tuning_objective_metric' in hyperparams:
        del hyperparams['_tuning_objective_metric']

    model = env.common_initialization(entry, Path('/opt/ml/code/package.tar.gz'),{}, 'model')
    Logger.info('Starting training now...')
    try:
        model.run_with_environment('/opt/ml/input/data/training/', env)
    except:
        Logger.info(traceback.format_exc())
        raise

#

