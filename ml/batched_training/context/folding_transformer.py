import pandas as pd


class FoldingTransformer:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = list(df.columns)
        names = df.index.names
        offset_name = names[1]
        df = df.reset_index()
        positive = df.loc[df[offset_name]>=0].copy()
        negative = df.loc[df[offset_name]<=0].copy()
        negative[offset_name] = negative[offset_name].abs()
        positive = positive.rename(columns={k:k+'_positive' for k in columns})
        negative = negative.rename(columns={k:k+'_negative' for k in columns})
        positive = positive.set_index(names)
        negative = negative.set_index(names)
        new_df = positive.merge(negative, left_index=True, right_index=True, how='outer').fillna(0)
        return new_df


