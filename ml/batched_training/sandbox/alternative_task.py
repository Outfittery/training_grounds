from ... import batched_training as bt
from .. import context as btc
from .. import torch as btt
from .. import gorynych as btg
from ... import dft
from .plain_context_builder import PlainContextBuilder
from sklearn.metrics import roc_auc_score
import torch

class AlternativeNetwork(torch.nn.Module):
    def __init__(self, sample, head: torch.nn.Module):
        super().__init__()
        self.head = head
        tensor = self.head(sample)
        self.linear = torch.nn.Linear(tensor.shape[1], 1)

    def forward(self, x):
        x = self.head(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x



class AlternativeTrainingTask(btt.TorchTrainingTask):
    def __init__(self):
        super(AlternativeTrainingTask, self).__init__()
        self.settings.mini_batch_size = 200
        self.settings.mini_epoch_count = 1
        self.metric_pool = bt.MetricPool().add_sklearn(roc_auc_score)
        self.tail_size = 50
        self.context_size = 15
        self.factory = btg.Gorynych()
        self.enabled_extractors = None

    def initialize_task(self, idb: bt.IndexedDataBundle):
        label_extractor = (
            bt.PlainExtractor
            .build(btt.Conventions.LabelFrame)
            .index()
            .apply(
                take_columns='label',
                transformer=None
            )
        )
        stem_extractor = btc.NewEmbeddingExtractor(
            bt.PlainExtractor.build('stem_embedding').index().join('snowball','another_word_id').apply(take_columns='stem', transformer=None),
            300
        )
        ending_extractor = btc.NewEmbeddingExtractor(
            bt.PlainExtractor.build('ending_embedding').index().join('snowball','another_word_id').apply(take_columns='ending', transformer=None),
            10000
        )
        feature_extractor = bt.PlainExtractor.build('features').index().join('pymorphy', 'another_word_id').apply(transformer=dft.DataFrameTransformerFactory.default_factory())

        inner_extractors = [stem_extractor, ending_extractor, feature_extractor]
        if self.enabled_extractors is not None:
            inner_extractors = [e for e in inner_extractors if e.get_name() in self.enabled_extractors]


        context_builder = PlainContextBuilder(include_zero_offset=True,left_to_right_contexts_proportion=0.5)

        self.context_extractor = self.factory.create_context_extractor_from_inner_extractors(
            'context_features',
            inner_extractors,
            context_builder
        )

        extractors = [self.context_extractor, label_extractor]

        self.setup_batcher(idb, extractors)
        self.setup_model(self.create_network, True)

    def create_network(self, batch):
        head = self.factory.create_context_head(batch, self.context_extractor)
        return AlternativeNetwork(batch, head)
