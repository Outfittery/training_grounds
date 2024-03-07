from typing import *
from ..._common import Logger
from dataclasses import dataclass
from pathlib import Path
import pickle
import traceback

@dataclass
class EntryPoint:
    name: str
    version: str
    job_location: Optional[Path]
    custom_job: Optional = None

    def _run_function(self, function: Callable):
        try:
            function()
            Logger.info('Job has exited successfully')
        except:
            tb = traceback.format_exc()
            Logger.error('Job has NOT exited sucessfully')
            Logger.error(tb)
            raise

    def run(self):
        if self.custom_job is None:
            try:
                with open(self.job_location,'rb') as file:
                    job = pickle.load(file)
                    Logger.info('Job of type ' + str(type(job)) + ' is loaded')
            except:
                tb = traceback.format_exc()
                Logger.error('Job is NOT loaded')
                Logger.error(tb)
                raise
        else:
            job = self.custom_job

        if callable(job):
            Logger.info('Job is callable, calling directly')
            self._run_function(job)
        elif hasattr(job, 'run'):
            Logger.info('Job has `run` attribute')
            if hasattr(job, 'set_calling_entry_point'):
                job.set_calling_entry_point(self)
            self._run_function(job.run)
        else:
            raise ValueError('Job is not callable and does not have `run` attributes')

        Logger.info('DONE. Exiting Training Grounds.')