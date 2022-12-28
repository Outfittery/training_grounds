from typing import *
from .perceptron import Perceptron
from .feed_forward_network import FeedForwardNetwork
from .input_conversion_network import InputConversionNetwork
from functools import partial

class Factories:
    class Tailing:
        def __init__(self, head_factory, output_frame_name):
            self.head_factory = head_factory
            self.output_frame_name = output_frame_name

        def __call__(self, sample):
            head = self.head_factory(sample)
            output_size = Perceptron.tensor_argument_to_int(sample[self.output_frame_name])
            head_output_size = head(sample)
            return FeedForwardNetwork(head, Perceptron(head_output_size, output_size))

    @staticmethod
    def FullyConnected(
            sizes: Union[Iterable[int], int],
            input_frame_name: Union[None, str, Iterable[str]] = None,
            output_frame_name: Union[None, str] = None):
        factories = []

        if input_frame_name is not None:
            factories.append(InputConversionNetwork(input_frame_name))

        if isinstance(sizes, int):
            sizes = [sizes]
        for size in sizes:
            factories.append(partial(Perceptron, output_size=size))

        factory = FeedForwardNetwork.Factory(*factories)
        if output_frame_name is None:
            return factory
        return Factories.Tailing(factory, output_frame_name)


