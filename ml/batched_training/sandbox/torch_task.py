from ... import batched_training as bt
from ... import dft
from .. import torch as btt
import torch

class SandboxNetwork(torch.nn.Module):
    def __init__(self, input_name, sizes):
        super().__init__()
        self.sizes = sizes
        self.input_name = input_name
        self.linears = torch.nn.ModuleList()
        for i in range(len(self.sizes)-1):
            self.linears.append(torch.nn.Linear(sizes[i], sizes[i+1]))

    def forward(self, input):
        x = btt.DfConversion.float(input[self.input_name])
        for layer in self.linears:
            x = layer(x)
            x = torch.sigmoid(x)
        return x



class SandboxTorchTask(btt.TorchTrainingTask):
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


    def network_factory(self, sample):
        all_sizes = [sample[self.input_frame_name].shape[1]]
        all_sizes.extend(self.network_sizes)
        all_sizes.append(sample[btt.Conventions.LabelFrame].shape[1])
        return SandboxNetwork(self.input_frame_name, all_sizes)

    def initialize_task(self, data):
        self.setup_batcher(data, self.extractors)
        self.setup_model(self.network_factory, ignore_consistancy_check=True)



