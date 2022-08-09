# 4.3. Training Jobs and Sagemaker (tg.common.delivery.training)

## Preparing the training task and data

Another scenario for delivery is `Sagemaker` training, that is applicable to the descendants of `TrainingTask`. We will demonstrate it with `SingleFrameTrainingTask`, as it has simpler setup, and titanic dataset iris dataset. 

First, we need to create a dataset and place it in the right folder.


```python
from sklearn import datasets
import pandas as pd
from pathlib import Path
import os

df = pd.read_csv('titanic.csv')
df = df.set_index('PassengerId')
for c in ['Pclass','SibSp','Parch','Survived']:
    df[c] = df[c].astype(float)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = df[features+['Survived']]
folder = Path('temp/datasets/titanic')
os.makedirs(folder, exist_ok=True)
df.to_parquet(folder/'titanic.parquet')
```

We will store it locally. We will not actually run this task on the `Sagemaker`, hence, there is no need to upload it. In real setup, you would need to upload the dataset to your `[bucket]`, respecting the following convention:

* Datasets are uploaded to `[bucket]/sagemaker/[project_name]/datasets/`
* Output of the training jobs is placed to `[bucket]/sagemaker/[project_name]/output`


```python
from tg.common.ml import single_frame_training as sft
from tg.common.ml import dft
from sklearn.metrics import roc_auc_score

task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider=sft.ModelProvider(sft.ModelConstructor(
            'sklearn.linear_model:LogisticRegression',
            max_iter = 1000),
        transformer = dft.DataFrameTransformerFactory.default_factory(),
        keep_column_names=False),
    evaluator=sft.Evaluation.binary_classification,
    splitter=sft.FoldSplitter(),
    metrics_pool = sft.MetricPool().add_sklearn(roc_auc_score)        
    )
```

To start Sagemaker training even on the local machine, one needs `AWS_ROLE`. We will import it from `environment.env` file:


```python
from tg.common import Loc
import dotenv

dotenv.load_dotenv(Loc.root_path/'environment.env')
'AWS_ROLE' in os.environ
```




    True




```python
from tg.common.delivery.training import SagemakerTrainingRoutine
from tg.common.delivery.packaging import FakeContainerHandler


routine = SagemakerTrainingRoutine(
    local_dataset_storage = Path('temp/datasets'),
    project_name = 'titanic',
    handler_factory = FakeContainerHandler.Factory(),
    aws_role = os.environ['AWS_ROLE'],
    s3_bucket = None
)
```

As with `SSHDockerRoutine`, there are `attached`, `local` and `remote` accessors. 

## `attached` accesor


```python
attached_task_id = routine.attached.execute(task,'titanic')
```

    2022-08-09 09:31:45.922679+00:00 INFO: Starting stage 1/1
    2022-08-09 09:31:46.098582+00:00 INFO: Saved artifact /home/yura/Desktop/repos/lesvik-ml/temp/training_results/_20220809_113145_650f41a138cb42b7897e252467ba98a2/runs/0/result_df
    2022-08-09 09:31:46.101082+00:00 INFO: Saved artifact /home/yura/Desktop/repos/lesvik-ml/temp/training_results/_20220809_113145_650f41a138cb42b7897e252467ba98a2/runs/0/metrics
    2022-08-09 09:31:46.103845+00:00 INFO: Saved artifact /home/yura/Desktop/repos/lesvik-ml/temp/training_results/_20220809_113145_650f41a138cb42b7897e252467ba98a2/runs/0/info
    2022-08-09 09:31:46.109628+00:00 INFO: Saved artifact /home/yura/Desktop/repos/lesvik-ml/temp/training_results/_20220809_113145_650f41a138cb42b7897e252467ba98a2/runs/0/model
    2022-08-09 09:31:46.112319+00:00 INFO: Saved artifact /home/yura/Desktop/repos/lesvik-ml/temp/training_results/_20220809_113145_650f41a138cb42b7897e252467ba98a2/runs/0/training_task
    2022-08-09 09:31:46.115990+00:00 INFO: Saved artifact /home/yura/Desktop/repos/lesvik-ml/temp/training_results/_20220809_113145_650f41a138cb42b7897e252467ba98a2/runs/0/train_split
    2022-08-09 09:31:46.117306+00:00 INFO: Saved artifact /home/yura/Desktop/repos/lesvik-ml/temp/training_results/_20220809_113145_650f41a138cb42b7897e252467ba98a2/runs/0/test_splits
    2022-08-09 09:31:46.119063+00:00 INFO: Completed stage 1/1
    2022-08-09 09:31:46.123981+00:00 INFO: ###METRIC###roc_auc_score_test:0.8538095238095237###
    2022-08-09 09:31:46.124788+00:00 INFO: ###METRIC###roc_auc_score_train:0.8600247283139194###


Unlike `SSHDockerRoutine`, `SagemakerTrainingRoutine` has the output, and `local` and `attached` accessors try to emulate `Sagemaker` behaviour in how the output is handled. They store the output in `Loc.temp` folder, and `execute` method returns a task id to access the result. Let's browse the result.


```python
from yo_fluq_ds import Query, FileIO

attached_folder = Loc.temp_path/'training_results'/attached_task_id
Query.folder(attached_folder, '**/*').foreach(lambda z: print(z.relative_to(attached_folder)))
```

    runs
    runs/0
    runs/0/metrics.pkl
    runs/0/info.pkl
    runs/0/train_split.pkl
    runs/0/result_df.parquet
    runs/0/test_splits.pkl
    runs/0/model.pkl
    runs/0/training_task.pkl


We can view the resulting dataframe, and compute, for instance, ROC AUC optimat threshold:


```python
from tg.common.ml.miscellaneous import roc_optimal_threshold

df = pd.read_parquet(attached_folder/'runs/0/result_df.parquet')
roc_optimal_threshold(df.true, df.predicted)

```




    0.3483943081626346



We can also unpickle model or the whole training task:


```python
FileIO.read_pickle(attached_folder/'runs/0/training_task.pkl')
```




    <tg.common.ml.single_frame_training.training_task.SingleFrameTrainingTask at 0x7fae1831cbb0>



## `local` accesor

Now, let's run the task in `local` mode, i.e. inside the docker container, but on the local machine:


```python
from IPython.display import clear_output

local_task_id = routine.local.execute(task,'titanic')
clear_output()
```

The output from `local` training is even closer to the real Sagemaker output: the model is packaged in `.tar.gz` file.


```python
local_task_id
```




    '_20220809_113146_91f12a755f194b9f9f4e3bad6ef7913e'




```python
local_path = Loc.temp_path/'training_results'/local_task_id
Query.folder(local_path).foreach(lambda z: print(z.relative_to(local_path)))
```

    model.tar.gz
    output.tar.gz


We have `open_sagemaker_result` method that will extract files from the archive and return `ResultPickleReader` instance.


```python
from tg.common.delivery.training import open_sagemaker_result

reader = open_sagemaker_result(local_path/'model.tar.gz', local_task_id)
```

    package.tag.gz
    task.json
    package.json
    hyperparameters.json
    runs/
    runs/0/
    runs/0/info.pkl
    runs/0/metrics.pkl
    runs/0/model.pkl
    runs/0/result_df.parquet
    runs/0/test_splits.pkl
    runs/0/train_split.pkl
    runs/0/training_task.pkl


From this `reader`, we can get the paths to the files and open them directly:


```python
df = pd.read_parquet(reader.get_path('runs/0/result_df.parquet'))
roc_optimal_threshold(df.true, df.predicted)
```




    0.3483943081626341



However, we cannot just open the `training_task`:


```python
import traceback
try:
    FileIO.read_pickle(reader.get_path('runs/0/training_task.pkl'))
except:
    print(traceback.format_exc())
```

    Traceback (most recent call last):
      File "/tmp/ipykernel_11999/225246354.py", line 3, in <module>
        FileIO.read_pickle(reader.get_path('runs/0/training_task.pkl'))
      File "/home/yura/anaconda3/envs/lesvik/lib/python3.8/site-packages/yo_fluq_ds/_misc/io.py", line 17, in read_pickle
        return pickle.load(file)
    ModuleNotFoundError: No module named 'titanic___20220809_113146_91f12a755f194b9f9f4e3bad6ef7913e'
    


Why? Because when delivering, we run all the packaging procedures, and those include creating a Training Grounds package with a unique id, and translating all the classes into this package. This package is available in the Docker container, but is not available in the python environment of the notebook where we're trying to read the results. Consequently, the reading fails.

Fortunately, `ResultPickleReader` contains a method that translates everything back:


```python
task = reader.unpickle('runs/0/training_task.pkl')
type(task)
```




    tg.common.ml.single_frame_training.training_task.SingleFrameTrainingTask



## Notes on `remote` accesor

In general, `remote` accesor performs the same way as `local`, but there are several important differences:
* `execute` has `wait` method. When set to `False`, it will trigger the process on Sagemaker servers and exits `execute` method immediately after the process has started, without waiting for it to end. This will allow you to run several tasks. If you choose to leave `wait` to `True`, you can terminate the process on your machine once the remote training has started, it will not affect the training at `Sagemaker` servers.
* instead of `open_sagemaker_result`, you may use `download_and_open_sagemaker_result`.



## Automatic task name's assignment

When multiple tasks are running, it's quite handy to assign to each a name that would represent the parameters of the task. Out initial idea was to implement this logic inside the task, but the downside of this approach is that parameters are many, while length of the task's name in Sagemaker is limited, and quickly reached. 

The alternative solution is to, first, use a factory method that builds tasks:


```python
from yo_fluq_ds import Obj

def build(
    learning_rate=1, 
    network_size=[10,10], 
    context_length = 10,
):
    return Obj(info=dict(name=''))
```

This `build` method returns a mock for training task: we are now interested only in `info` field of the task, that will contain the name.


```python
from tg.common.delivery.training import Autonamer

Autonamer(build).build_tasks(learning_rate = [1, 2], network_size = [[10], [10, 5]])
```




    [{'info': {'name': 'LR1-NS10'}},
     {'info': {'name': 'LR1-NS10-5'}},
     {'info': {'name': 'LR2-NS10'}},
     {'info': {'name': 'LR2-NS10-5'}}]



As we can see, `Autonamer` will instantiate all the tasks and assign automatically generated names to them. Note that it does not create entry for `context_length` in the name, as it is not variable in this run. 


