from typing import *
from .. import torch as btt
from .. import context as btc
from ... import batched_training as bt

class TensorlikeExtractorToAggregator(btc.ExtractorToAggregatorFactory):
    def __init__(self,
                 extractors: Iterable[bt.Extractor],
                 extrator_name_to_conversion: Optional[Dict[str, Callable]],
                 reverse_context_order: bool = False,
                 feature_transformer: Optional = None
                 ):
        self.extractors = extractors
        self.extractor_name_to_conversion = extrator_name_to_conversion if extrator_name_to_conversion is not None else {}
        self.reverse_context_order = reverse_context_order
        self.feature_transformer = feature_transformer

    def create_extractors_and_aggregators(self, ibundle: bt.IndexedDataBundle) -> List[btc.ExtractorToAggregator]:
        result = []
        for ex in self.extractors:
            ex.fit(ibundle)
            input = ex.extract(ibundle)
            conversion = btt.DfConversion.auto
            if ex.get_name() in self.extractor_name_to_conversion:
                conversion = self.extractor_name_to_conversion[ex.get_name()]
            aggregator = btc.LSTMAggregator(
                conversion=conversion,
                feature_transformer=self.feature_transformer,
                reverse_context_order=self.reverse_context_order
            )
            aggregator.fit(ibundle.index_frame, input)
            result.append(btc.ExtractorToAggregator(ex, [aggregator]))
        return result