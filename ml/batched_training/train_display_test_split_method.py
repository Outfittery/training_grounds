from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def _train_display_test_split_by_group(df, group_column, test_size=0.3, display_size=0.3):
    values = df[group_column].unique()
    train, test = train_test_split(values, test_size=test_size)
    train, display = train_test_split(train, test_size=display_size/(1-test_size))
    split = np.where(
        df[group_column].isin(test),
        'test',
        np.where(df[group_column].isin(display),
                 'display',
                 'train'))
    return pd.Series(split, index=df.index)


def train_display_test_split(
        df,
        test_size=0.3,
        display_size=0.3,
        stratify_column = None,
        group_column = None
):
    if group_column is not None:
        return _train_display_test_split_by_group(df, group_column, test_size, display_size)

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