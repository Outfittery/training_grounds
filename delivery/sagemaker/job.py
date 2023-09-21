from typing import *
from ..._common import Logger
from ..delivery import EntryPoint
from .environment import SagemakerEnvironment
from pathlib import Path
from yo_fluq_ds import FileIO
import json
import shutil
import jsonpickle
import traceback

from yo_fluq_ds import Query

class SagemakerJob:
    def __init__(self, task):
        self.task = task
        self.entry_point = None #type: Optional[EntryPoint]

    def set_calling_entry_point(self, entry_point: EntryPoint):
        self.entry_point = entry_point

    def process_package(self):
        Logger.info('Preparing package properties...')
        package_props = self.entry_point.get_properties()
        props_str = json.dumps(package_props)
        Logger.info(props_str)
        FileIO.write_text(props_str, str(self.folder / 'package.json'))

        Logger.info('Preparing package file...')
        package_location = Path('/opt/ml/code/package.tar.gz')
        shutil.copy(str(package_location), str(self.folder / 'package.tag.gz'))

    def process_hyperparameters(self):
        Logger.info('Processing hyperparameters...')
        hyperparams = FileIO.read_json('/opt/ml/input/config/hyperparameters.json')
        if hyperparams is None:
            hyperparams = {}
        if '_tuning_objective_metric' in hyperparams:
            del hyperparams['_tuning_objective_metric']
        if len(hyperparams)==0:
            Logger.info("No hyperparameters are provided")
        else:
            Logger.info("Hyperparameters are provided")
            Logger.info(hyperparams)
            Logger.info("Storing Hyperparameters in file")
            FileIO.write_json(hyperparams, str(self.folder/'hyperparameters.json'))
            Logger.info(f'Applying hyperparams')
            self.task.apply_hyperparams(hyperparams)

        Logger.info("Model initialized. Jsonpickling...")
        model_state = json.dumps(json.loads(jsonpickle.dumps(self.task)), indent=1)
        FileIO.write_text(model_state, str(self.folder/'task.json'))

    def try_restore_from_checkpoint(self):
        if not self.checkpoint_folder.is_dir():
            return False
        #TODO: will only work for the batched training.
        #Future notes: the task (which is known at this point, because it was packaged with SagemakerJob)
        #Should report which is the file to restore it from

        file_path = self.checkpoint_folder/'output/training_task.pkl'
        try:
            self.task = FileIO.read_pickle(file_path)
            self.task.settings.continue_training = True #TODO: Again, normally, this should happen in the task itself after initialization
            Logger.info(f'Task successfully restored from {file_path}')

            # copy checkpoint to output so if the task is interrupted before the first epoch ends, the results are still available
            shutil.copytree(self.checkpoint_folder, self.folder, dirs_exist_ok=True)
            return True
        except:
            Logger.warning(f'Task was not successfully restored')
            Logger.warning(traceback.format_exc())
            return False


    def run(self):
        Logger.info('This is Sagemaker Job performing a training task')
        self.folder = Path('/opt/ml/model')
        self.checkpoint_folder = Path('/opt/checkpoint')

        env = SagemakerEnvironment(self.folder, self.checkpoint_folder)
        self.process_package()

        if not self.try_restore_from_checkpoint():
            Logger.info('New training')
            self.process_hyperparameters()
        else:
            Logger.info('Continuing training from the checkpoint')

        Logger.info('Starting training now...')
        try:
            self.task.run_with_environment('/opt/ml/input/data/training/', env)
        except:
            Logger.error(traceback.format_exc())
            raise