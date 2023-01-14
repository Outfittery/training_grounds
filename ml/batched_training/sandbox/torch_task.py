from ... import batched_training as bt
from ... import dft
from .. import factories as btf


class SandboxTorchTask(btf.TorchTrainingTask):
    def __init__(self,
                 extractors,
                 network_sizes,
                 metric,
                 input_frame_name = 'features',
                 ):
        super(SandboxTorchTask, self).__init__()
        self.extractors = extractors
        self.network_sizes = network_sizes
        self.input_frame_name = input_frame_name
        if isinstance(metric, bt.Metric):
            self.metric_pool = bt.MetricPool().add(metric)
        else:
            self.metric_pool = bt.MetricPool().add_sklearn(metric)


    def initialize_task(self, data):
        self.setup_batcher(data, self.extractors)
        network_factory = btf.Factories.Tailing(
            btf.Factories.FullyConnected(self.network_sizes, self.input_frame_name),
            btf.Conventions.LabelFrame
        )
        self.setup_model(network_factory, ignore_consistancy_check=True)



