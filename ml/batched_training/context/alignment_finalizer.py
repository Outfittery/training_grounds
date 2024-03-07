from typing import *
from .architecture import AggregationFinalizer
from ..torch import AnnotatedTensor
import pandas as pd

class AlignmentAggregationFinalizer(AggregationFinalizer):
    def __init__(self, use_dict_if_one_tensor: bool = False):
        self.use_dict_if_one_tensor = use_dict_if_one_tensor
        pass

    def finalize(self, index: pd.DataFrame, features: Dict[str, pd.DataFrame], aggregations: Dict[str, pd.DataFrame]):
        if len(features) == 0:
            raise ValueError('No features were produced')
        tensors = [] #type: List[AnnotatedTensor]
        keys = []
        for key, value in aggregations.items():
            if not isinstance(value, AnnotatedTensor):
                raise ValueError(f"Aggregation `{key}` is supposed to be Annotated Tensor, but was `{type(value)}")
            tensors.append(value)
            keys.append(key)


        if not self.use_dict_if_one_tensor and len(tensors) == 1:
            return tensors[0]

        for i in range(1, len(tensors)):
            if tensors[0].dim_names != tensors[i].dim_names:
                raise ValueError(f"Alignment failed at {keys[i]}: dim_names are {tensors[i].dim_names}, expected {tensors[0].dim_names}")
            for j in range(2):
                if tensors[0].dim_indices[j] != tensors[i].dim_indices[j]:
                    raise ValueError(f"Alignment failed at {keys[i]}: indices {j} are different. Actual\n{tensors[i].dim_indices[j]}\nExpected\n{tensors[i].dim_indices[j]}")
        return {key: value for key, value in zip(keys, tensors)}




