from ... import batched_training as bt
from ... import dft
from .. import factories as btf
from sklearn import datasets
import pandas as pd

def get_binary_classification_bundle():
    ds = datasets.load_breast_cancer()
    features = pd.DataFrame(ds['data'], columns=ds['feature_names'])
    df = pd.DataFrame(ds['target'], columns=['label'])
    df['split'] = bt.train_display_test_split(df, 0.2, 0.2, 'label')
    bundle = bt.DataBundle(index=df, features=features)
    return bundle


def get_multilabel_classification_bundle():
    ds = datasets.load_iris()
    features = pd.DataFrame(ds['data'], columns=ds['feature_names'])
    df = pd.DataFrame(ds['target_names'][ds['target']], columns = ['label'])
    df['split'] = bt.train_display_test_split(df, 0.2, 0.2, 'label')
    bundle = bt.DataBundle(index=df, features=features)
    return bundle


def get_feature_extractor():
    feature_extractor = (bt.PlainExtractor
                         .build('features')
                         .index('features')
                         .apply(transformer=dft.DataFrameTransformerFactory.default_factory())
                         )
    return feature_extractor


def get_binary_label_extractor():
    label_extractor = (bt.PlainExtractor
                       .build(btf.Conventions.LabelFrame)
                       .index()
                       .apply(take_columns=['label'], transformer=None)
                       )
    return label_extractor


def get_multilabel_extractor():
    label_extractor = (bt.PlainExtractor
                   .build(btf.Conventions.LabelFrame)
                   .index()
                   .apply(take_columns=['label'], transformer=dft.DataFrameTransformerFactory.default_factory())
                  )
    return label_extractor
