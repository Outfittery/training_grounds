from typing import *
import torch
import pandas as pd
from .basics import AnnotatedTensor
from ....._common import DataBundle


class InputConversionNetwork(torch.nn.Module):
    def __init__(self,
                 input_frames: Union[None, str, Iterable[str]],
                 raise_if_inputs_are_missing=False
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

    @staticmethod
    def df_to_torch(df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(df.astype(float).values).float()

    @staticmethod
    def collect_tensors(input, input_frames, raise_if_inputs_are_missing):
        en = input_frames
        tensors = []
        for frame in en:
            if frame not in input.bundle.data_frames:
                if raise_if_inputs_are_missing:
                    raise ValueError(f'Missing frame {frame} in batch')
                else:
                    continue
            if isinstance(input[frame], pd.DataFrame):
                tensors.append(InputConversionNetwork.df_to_torch(input[frame]))
            elif isinstance(input[frame], AnnotatedTensor):
                tensors.append(input[frame].tensor)
            elif isinstance(input[frame], torch.Tensor):
                tensors.append(input[frame])
            else:
                raise ValueError(
                    f'Batch element must be torch.Tensor, pandas.Dataframe or AnnotatedTensor, but was {type(input[frame])}')

        return tensors

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            if self.input_frames is not None:
                raise ValueError('The input was tensor, but `input_frames` were provided, suggesting it would be a dictionary')

        if isinstance(input, DataBundle):
            input = input.data_frames

        if self.input_frames is None:
            en = list(input.keys())
        else:
            en = self.input_frames
        tensors = InputConversionNetwork.collect_tensors(input, en, self.raise_if_inputs_are_missing)

        if len(tensors) == 0:
            raise ValueError(f'No tensors were produced. Input keys are {list(input.keys())}, expected keys are {list(en)}')
        if len(tensors) == 1:
            return tensors[0]
        else:
            return torch.cat(tensors, 1)


