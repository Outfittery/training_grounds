from ....ml.single_frame_training import *
from sklearn import datasets
import shutil

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

def create_dataset_files(path, test_name):
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    os.makedirs(path/test_name)
    df = get_data()
    df.to_parquet(path/test_name/'data.parquet')




