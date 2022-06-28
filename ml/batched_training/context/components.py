from .architecture import *


class PandasAggregationFinalizer(AggregationFinalizer):
    def __init__(self,  add_presence_columns: bool = True):
        self.add_presence_columns = add_presence_columns

    def _compose_columns(self, index_frame: pd.DataFrame, agg_results: Dict[str, pd.DataFrame]):
        result = {}
        for agg_name, agg in agg_results.items():
            for c in agg.columns:
                result[f'{agg_name}_{c}'] = agg[c]
            if self.add_presence_columns:
                result[f'{agg_name}_present_{agg_name}'] = pd.Series(1, index=agg.index)
        return result

    def fit(self, index: pd.DataFrame, features: Dict[str, pd.DataFrame], aggregations: Dict[str,pd.DataFrame]):
        self.columns_ = list(self._compose_columns(index, aggregations))

    def _compute_intermediate_dict(self, index, aggregations):
        inner_results = self._compose_columns(index, aggregations)
        result = {}
        for c in self.columns_:
            if c in inner_results:
                value = inner_results[c]
                if not isinstance(value, pd.Series):
                    raise ValueError(f'Something wrong is happened around column {c}, the type of extraction {type(value)} (duplicating column?)')
                result[c] = value
            else:
                result[c] = pd.Series(0, index=index.index)
        return result

    def finalize(self, index: pd.DataFrame, features: Dict[str, pd.DataFrame], aggregations: Dict[str, pd.DataFrame]):
        result = self._compute_intermediate_dict(index, aggregations)
        rdf = pd.DataFrame(result)
        rdf = index[[]].merge(rdf,left_index=True, right_index=True, how='left')
        rdf = rdf.fillna(0)
        rdf = rdf.loc[index.index]
        rdf = rdf[self.columns_]
        return rdf


class GroupByAggregator(ContextAggregator):
    def __init__(self, methods):
        self.methods = methods

    def aggregate_context(self, features_df):
        names = features_df.index.names
        if names[0] is None:
            raise ValueError('There is `None` in the features df index. This aggregator requires you to set the name for index of your samples')
        columns = features_df.columns
        df = features_df.reset_index()
        df = df.groupby(names[0])[columns].aggregate(self.methods)
        df.columns = [f'{a}_{b}' for a, b in df.columns]
        return df


class PivotAggregator(ContextAggregator):
    def __init__(self, presence_column = False):
        self.presence_column = presence_column



    def aggregate_context(self, features_df):
        names = features_df.index.names
        if names[0] is None:
            raise ValueError('There is `None` in the features df index. This aggregator requires you to set the name for index of your samples')
        columns = list(features_df.columns)
        df = features_df.reset_index()
        if self.presence_column:
            df = df.assign(offset_is_presenting = 1)
            columns.append('offset_is_presenting')
        df = df.pivot_table(index=names[0], columns=names[1], values=columns).fillna(0)
        df.columns = [f'{a}_at_{b}' for a, b in df.columns]
        return df


