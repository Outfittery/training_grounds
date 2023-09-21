from tg.common.ml import batched_training as bt
from .. import context_2 as btc
from .. import factories as btf
from ... import dft
from .plain_context_builder import PlainContextBuilder
from sklearn.metrics import roc_auc_score

def create_assembly_point() -> btc.ContextAssemblyPoint2:
    units = []

    stem_extractor = bt.PlainExtractor.build('stem_embedding').index().join('snowball','another_word_id').apply(take_columns='stem', transformer=None)
    units.append(btc.NewEmbeddingAssemblyUnit(stem_extractor, vocab_size=100000, dimensions=20))

    ending_extractor = bt.PlainExtractor.build('ending_embedding').index().join('snowball','another_word_id').apply(take_columns='ending', transformer=None)
    units.append(btc.NewEmbeddingAssemblyUnit(ending_extractor, vocab_size=500, dimensions=20))

    feature_extractor = bt.PlainExtractor.build('features').index().join('pymorphy', 'another_word_id').apply(transformer=dft.DataFrameTransformerFactory.default_factory())
    units.append(btc.FeaturesAssemblyUnit(feature_extractor))

    context_builder = PlainContextBuilder(include_zero_offset=True,left_to_right_contexts_proportion=0.5)
    return btc.ContextAssemblyPoint2(
        units,
        context_builder,
    )

class AlternativeTrainingTask2(btf.TorchTrainingTask):
    def __init__(self):
        super(AlternativeTrainingTask2, self).__init__()
        self.settings.mini_batch_size = 200
        self.settings.mini_epoch_count = 1

        self.assembly_point = create_assembly_point()

        self.metric_pool = bt.MetricPool().add_sklearn(roc_auc_score)

        self.tail_size = 50

    def initialize_task(self, idb: bt.IndexedDataBundle):
        label_extractor = (
            bt.PlainExtractor
            .build(btf.Conventions.LabelFrame)
            .index()
            .apply(
                take_columns='label',
                transformer=None
            )
        )
        extractors = [
            self.assembly_point.create_extractor(),
            label_extractor
        ]

        self.setup_batcher(idb, extractors)

        head_factory = self.assembly_point.create_network_factory()
        factory = btf.FeedForwardNetwork.Factory(
            head_factory,
            btf.Factories.Factory(btf.Perceptron, output_size=self.tail_size),
            btf.Factories.Factory(btf.Perceptron, output_size=1)
        )
        self.setup_model(factory, True)