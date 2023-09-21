from ..extractors import Extractor, IndexedDataBundle
from ..factories import Factories, InputConversionNetwork, AssemblyPoint
from copy import deepcopy
import pandas as pd

class FeaturesAssemblyUnit(AssemblyPoint):
    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    def get_name(self):
        return self.extractor.get_name()

    def create_extractor(self):
        return deepcopy(self.extractor)

    def create_network_factory(self):
        return Factories.Factory(InputConversionNetwork(self.extractor.get_name()))



