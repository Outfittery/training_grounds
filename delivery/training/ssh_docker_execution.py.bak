import logging
import os

from pathlib import Path

from .architecture import FileCacheTrainingEnvironment
from ..packaging import EntryPoint



logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger()



def execute(entry: EntryPoint):
    folder = Path('/opt/ml/model')
    os.makedirs(folder, exist_ok=True)
    env = FileCacheTrainingEnvironment(logger.info, folder, Path('/home/output/result.tar.gz'))
    model = env.common_initialization(entry, Path('/featurization/package.tar.gz'),{}, 'job')
    env.log('Starting training now...')
    model.run_with_environment('/home/data', env)

