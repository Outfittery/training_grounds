import pandas as pd
import torch

'''
                  embedding
offset sample_id           
-4     15                 2
       23                 0
-3     15                 3
       23                 0
-2     15                 0
       23                 1
-1     15                 0
       23                 2
embedding    int32
dtype: object
embedding    True
dtype: bool
'''

class DfConversion:
    @staticmethod
    def float(df: pd.DataFrame):
        return torch.tensor(df.astype(float).values).float()


    @staticmethod
    def int(df: pd.DataFrame):
        return torch.tensor(df.astype(int).values)

    @staticmethod
    def auto(df: pd.DataFrame):
        if (df.dtypes.astype(str).isin(['int64','int32'])).all():
            return DfConversion.int(df)
        return DfConversion.float(df)
