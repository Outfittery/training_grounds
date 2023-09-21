from typing import *
from ..._common import DataBundle, Logger


class PredictionJob:
    def __init__(self,
                 bundle_source: Callable[[],DataBundle],
                 bundle_debug_storage: Optional[Callable[[DataBundle],None]],
                 predictor: Optional[Callable[[DataBundle], DataBundle]],
                 output_storage: Optional[Callable[[DataBundle], None]],
                 initialization: Optional[Callable] = None,
                 ):
        self.initialization = initialization
        self.bundle_source = bundle_source
        self.bundle_debug_storage = bundle_debug_storage
        self.predictor = predictor
        self.output_storage = output_storage


    def run(self):
        if self.initialization is not None:
            self.initialization()
        Logger.info("Getting bundle")
        bundle = self.bundle_source()
        if self.bundle_debug_storage is not None:
            Logger.info('Stashing bundle')
            self.bundle_debug_storage(bundle)
        if self.predictor is not None:
            Logger.info('Running predictor')
            result = self.predictor(bundle)
            if self.output_storage is not None:
                Logger.info('Uploading results')
                self.output_storage(result)