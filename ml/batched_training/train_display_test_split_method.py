from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def train_display_test_split(df, test_size=0.3, display_size=0.3, stratify_column = None):
    train, test = train_test_split(
        df.index,
        stratify=None if stratify_column is None else df[stratify_column],
        test_size=test_size)

    train, display = train_test_split(
        train,
        stratify=None if stratify_column is None else df.loc[train][stratify_column],
        test_size=display_size/(1-test_size))
    split = np.where(
        df.index.isin(test),
        'test',
        np.where(df.index.isin(display),
                 'display',
                 'train'))
    return pd.Series(split, index = df.index)