from typing import *
from .data_bundle import IndexedDataBundle

import pandas as pd


class BatchedModelHandler:
    """
    This is a bridge between a ``BatchedTrainingTask`` and pytorch model.
    """

    def instantiate(self, task, input: IndexedDataBundle) -> None:
        """
        Acts like a factory. Should instantiate the model and everything that is needed for it
        Args:
            task: TrainingTask that requests initialization
            input: non-label inputs
        """
        raise NotImplementedError()

    def train(self, input: IndexedDataBundle) -> float:
        """
        Trains model on the batch.
        Different training strategies (weights decay, simulated annealing) should be implemented here
        Args:
            input: all inputs, inluding labels
        Returns:
            Loss on the batch
        """
        raise NotImplementedError()

    def predict(self, input: IndexedDataBundle) -> pd.DataFrame:
        """
        Predicts the values for batch
        Args:
            input: all inputs, inluding labels

        Returns:
            Dataframe with predictions

        """
        raise NotImplementedError()
