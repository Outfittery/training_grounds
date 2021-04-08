from ....ml.single_frame_training import *
from sklearn import datasets

def create_task(max_iter=1000):
    return SingleFrameTrainingTask(
        data_loader = DataFrameLoader('target'),
        model_provider=ModelProvider(ModelConstructor(
                'sklearn.linear_model:LogisticRegression',
                max_iter = max_iter
            ),
            keep_column_names=False),
        evaluator=Evaluation.multiclass_classification,
        splitter=FoldSplitter(),
        with_tqdm=False
    )

def get_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df['target'] = iris['target']
    df = df[['target'] + iris['feature_names']]
    return df
