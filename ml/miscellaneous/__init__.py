import numpy as np
import pandas as pd



class QuantileProportionDifference:
    def __init__(self, lower_bound: float):
        self.lower_bound = lower_bound

    def __call__(self, true, predicted):
        lower = np.percentile(predicted, 100 * self.lower_bound)
        upper = np.percentile(predicted, 100 - 100 * self.lower_bound)
        true = pd.Series(true)
        predicted = pd.Series(predicted)
        lower_true = true.loc[predicted <= lower]
        upper_true = true.loc[predicted >= upper]
        return np.mean(upper_true) - np.mean(lower_true)
