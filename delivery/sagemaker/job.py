from ..._common import Logger
from ..delivery import EntryPoint
from .environment import SagemakerEnvironment
from pathlib import Path
from yo_fluq_ds import FileIO
import json
import shutil
import jsonpickle
import traceback

class SagemakerJob:
    def __init__(self, task):
        self.task = task
        self.entry_point = None

    def set_calling_entry_point(self, entry_point: EntryPoint):
        self.entry_point = entry_point

    def run(self):
        Logger.info('This is Sagemaker Job performing a training task')
        folder = Path('/opt/ml/model')
        env = SagemakerEnvironment(folder)

        Logger.info('Preparing package properties...')
        package_props = self.entry_point.get_properties()
        props_str = json.dumps(package_props)
        Logger.info(props_str)
        FileIO.write_text(props_str, str(folder/'package.json'))

        Logger.info('Preparing package file...')
        package_location = Path('/opt/ml/code/package.tar.gz')
        shutil.copy(str(package_location), str(folder/'package.tag.gz'))

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
            FileIO.write_json(hyperparams, str(folder/'hyperparameters.json'))
            Logger.info(f'Applying hyperparams')
            self.task.apply_hyperparams(hyperparams)

        Logger.info("Model initialized. Jsonpickling...")
        model_state = json.dumps(json.loads(jsonpickle.dumps(self.task)), indent=1)
        FileIO.write_text(model_state, str(folder/'task.json'))

        Logger.info('Starting training now...')
        try:
            self.task.run_with_environment('/opt/ml/input/data/training/', env)
        except:
            Logger.info(traceback.format_exc())
            raise