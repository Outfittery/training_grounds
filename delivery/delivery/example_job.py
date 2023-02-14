from ..._common import Logger
import os

class ExampleJob:
    def __init__(self, check_environmental_variables = None):
        if check_environmental_variables is None:
            self.check_environmental_variables = []
        else:
            self.check_environmental_variables = list(check_environmental_variables)


    def run(self):
        for env in self.check_environmental_variables:
            Logger.info(f'Variable {env} is found: {env in os.environ}')
        Logger.info('SUCCESS')
