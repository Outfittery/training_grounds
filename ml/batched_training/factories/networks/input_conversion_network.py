from typing import *
import torch
import pandas as pd
from .basics import AnnotatedTensor
from ...data_bundle import DataBundle, IndexedDataBundle
from ..conversion import DfConversion


class InputConversionNetwork(torch.nn.Module):
    def __init__(self,
                 input_frames: Union[None, str, Iterable[str]],
                 raise_if_inputs_are_missing=False,
                 conversion: Optional[Callable] = None
                 ):
        super(InputConversionNetwork, self).__init__()
        if input_frames is None:
            pass
        elif isinstance(input_frames, str):
            input_frames = [input_frames]
        elif iter(input_frames):
            input_frames = list(input_frames)
        else:
            raise ValueError(f'Expected str or Iterable[str] or None, got {type(input_frames)}, value {input_frames}')
        self.input_frames = input_frames
        self.raise_if_inputs_are_missing = raise_if_inputs_are_missing
        self.conversion = conversion
        if self.conversion is None:
            self.conversion = DfConversion.auto


    @staticmethod
    def collect_tensors(input, input_frames, raise_if_inputs_are_missing, conversion: Callable):
        en = input_frames
        tensors = []
        for frame in en:
            if frame not in input.bundle.data_frames:
                if raise_if_inputs_are_missing:
                    raise ValueError(f'Missing frame {frame} in batch')
                else:
                    continue
            if isinstance(input[frame], pd.DataFrame):
                tensors.append(conversion(input[frame]))
            elif isinstance(input[frame], AnnotatedTensor):
                tensors.append(input[frame].tensor)
            elif isinstance(input[frame], torch.Tensor):
                tensors.append(input[frame])
            else:
                raise ValueError(
                    f'Batch element must be torch.Tensor, pandas.Dataframe or AnnotatedTensor, but was {type(input[frame])}')

        return tensors

    def forward(self, input):
        if self.input_frames is None:
            en = list(input.keys())
        else:
            en = self.input_frames

        if isinstance(input, torch.Tensor):
            if self.input_frames is not None:
                raise ValueError('The input was tensor, but `input_frames` were provided, suggesting it would be an IndexedDataBundle')
        elif not isinstance(input, IndexedDataBundle):
            raise ValueError("Input expected to be tensor or `IndexedDataBundle`")

        tensors = InputConversionNetwork.collect_tensors(input, en, self.raise_if_inputs_are_missing, self.conversion)

        if len(tensors) == 0:
            raise ValueError(f'No tensors were produced. Expected keys are {list(en)}, input is {input}')
        if len(tensors) == 1:
            return tensors[0]
        else:
            return torch.cat(tensors, 1)


