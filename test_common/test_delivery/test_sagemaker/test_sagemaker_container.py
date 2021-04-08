from unittest import TestCase
from tg.common.delivery.sagemaker import *
from tg.common.ml.single_frame_training import *
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
from tg.common._common.locations import Loc


def get_iris_df():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df['target'] = iris['target']
    df = df[['target'] + iris['feature_names']]
    return df


def create_task(_, __):
    return SingleFrameTrainingTask(
        DataFrameLoader('target'),
        ModelProvider(ModelConstructor(
            'sklearn.linear_model:LogisticRegression',
        )),
        Evaluation.multiclass_classification,
        MetricPool().add_sklearn(accuracy_score).add_sklearn(f1_score),
        FoldSplitter(),

        with_tqdm=False
    )


class ContainerTestCase(TestCase):
    def test_container(self):
        tmp_path = Loc.temp_path.joinpath('tests/container_test').absolute()
        os.makedirs(tmp_path, exist_ok=True)
        train_path = tmp_path.joinpath('training/')
        os.makedirs(train_path, exist_ok=True)
        df = get_iris_df()
        basic_path = train_path.joinpath('basic')
        os.makedirs(basic_path, exist_ok=True)
        df.to_parquet(basic_path.joinpath('training.parquet'))

        uuid = '1dde9df4-b2df-4bd2-ac7c-75a2e30ed253'
        routine = SageMakerRoutine(
            None,
            'sagemaker_test',
            train_path,
            create_task,
            None,
            False
        )

        # result = routine.run_in_local_container({},'basic', tmp_path)
