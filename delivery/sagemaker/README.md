# 4.3. Training Jobs and Sagemaker (tg.common.delivery.sagemaker)

## Preparing the training task and data

Another scenario for delivery is `Sagemaker` training, that is applicable to the descendants of `TrainingTask`. We will demonstrate it with `SingleFrameTrainingTask`, as it has simpler setup, and titanic dataset. 

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
datasets_folder = Path('temp/datasets/titanic')
dataset_file = datasets_folder/'titanic_project/titanic.parquet'
os.makedirs(dataset_file.parent, exist_ok=True)
df.to_parquet(dataset_file)
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



Sagemaker delivery has a similar structure to the SSH/Docker: SagemakerOptions, SagemakerConfig, and three executors. As with SSH/Docker, the best way to use all this is to write a SagemakerRoutine which will set up all these.

Some notes before we start:
  * The task is not, by itself, a job, it is not self-contained, as the artefacts output is controlled by `TrainingEnvironment`. So, `SagemakerJob` is a job in the sence of `tg.common.delivery` that wraps the task and adopts its behaviour to sagemaker. Other cloud providers will probably required different tasks.
  * Sagemaker itself reqiures some specifics in container files, so this also needs to be reflected.
  * There are many dependencies required for training are, so we will need to change the default dependency lists.




```python
from tg.common.delivery.sagemaker import (SagemakerJob, SagemakerAttachedExecutor, SagemakerLocalExecutor, 
                                          DOCKERFILE_TEMPLATE, SagemakerOptions, SagemakerConfig)
from tg.common.delivery.delivery import Packaging, Containering, DependencyList
from yo_fluq_ds import *

dependencies = FileIO.read_json('dependencies.json')
dependencies = DependencyList('training', dependencies)


class SagemakerRoutine:
    def __init__(self,
                 task,
                 dataset: str,
                 project_name: str,
                 ):
        name = type(task).__name__
        task.info['name'] = name
        version = '0'
        job = SagemakerJob(task)
        packaging = Packaging(name, version, dict(job=job))
        packaging.dependencies = [dependencies]
        packaging.silent = True

        containering = Containering.from_packaging(packaging)
        containering.dependencies = [dependencies]
        containering.dockerfile_template = DOCKERFILE_TEMPLATE
        containering.run_file_name='train.py'
        containering.silent = True

        settings = SagemakerOptions(
            os.environ.get('AWS_ROLE'),
            None,
            project_name,
            datasets_folder,
            dataset,
        )

        self.config = SagemakerConfig(
            job,
            packaging,
            containering,
            settings
        )
        
    def attached(self):
        return SagemakerAttachedExecutor(self.config)
        
    def local(self):
        return SagemakerLocalExecutor(self.config)
    
routine = SagemakerRoutine(task,'titanic.parquet','titanic_project')
result = routine.attached().execute()
result['metrics']
```


      0%|          | 0/1 [00:00<?, ?it/s]


    2022-12-20 17:37:25.113050 INFO: Starting stage 1/1
    2022-12-20 17:37:25.233140 INFO: Completed stage 1/1
    2022-12-20 17:37:25.236670 INFO: ###roc_auc_score_test:0.8538095238095237
    2022-12-20 17:37:25.237406 INFO: ###roc_auc_score_train:0.8600247283139194





    {'roc_auc_score_test': 0.8538095238095237,
     'roc_auc_score_train': 0.8600247283139194}



Now we will run it in the local container:


```python
id = routine.local().execute()
```

    warning: no files found matching '*.yml' under directory 'SingleFrameTrainingTask__0'
    warning: no files found matching '*.rst' under directory 'SingleFrameTrainingTask__0'
    warning: sdist: standard file not found: should have one of README, README.rst, README.txt, README.md
    


    sha256:c5e1a98741882ce94e09ce458d7ace052fbdd2517633901743fe506d457c8cfd
    Creating zuve7o5y6g-algo-1-zipal ... 
    [1BAttaching to zuve7o5y6g-algo-1-zipal2mdone[0m
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:41,030 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:41,064 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:41,084 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:41,097 sagemaker-containers INFO     Invoking user script
    [36mzuve7o5y6g-algo-1-zipal |[0m 
    [36mzuve7o5y6g-algo-1-zipal |[0m Training Env:
    [36mzuve7o5y6g-algo-1-zipal |[0m 
    [36mzuve7o5y6g-algo-1-zipal |[0m {
    [36mzuve7o5y6g-algo-1-zipal |[0m     "additional_framework_parameters": {},
    [36mzuve7o5y6g-algo-1-zipal |[0m     "channel_input_dirs": {
    [36mzuve7o5y6g-algo-1-zipal |[0m         "training": "/opt/ml/input/data/training"
    [36mzuve7o5y6g-algo-1-zipal |[0m     },
    [36mzuve7o5y6g-algo-1-zipal |[0m     "current_host": "algo-1-zipal",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "framework_module": null,
    [36mzuve7o5y6g-algo-1-zipal |[0m     "hosts": [
    [36mzuve7o5y6g-algo-1-zipal |[0m         "algo-1-zipal"
    [36mzuve7o5y6g-algo-1-zipal |[0m     ],
    [36mzuve7o5y6g-algo-1-zipal |[0m     "hyperparameters": {},
    [36mzuve7o5y6g-algo-1-zipal |[0m     "input_config_dir": "/opt/ml/input/config",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "input_data_config": {
    [36mzuve7o5y6g-algo-1-zipal |[0m         "training": {
    [36mzuve7o5y6g-algo-1-zipal |[0m             "TrainingInputMode": "File"
    [36mzuve7o5y6g-algo-1-zipal |[0m         }
    [36mzuve7o5y6g-algo-1-zipal |[0m     },
    [36mzuve7o5y6g-algo-1-zipal |[0m     "input_dir": "/opt/ml/input",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "is_master": true,
    [36mzuve7o5y6g-algo-1-zipal |[0m     "job_name": "singleframetrainingtask-2022-12-20-16-37-38-243",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "log_level": 20,
    [36mzuve7o5y6g-algo-1-zipal |[0m     "master_hostname": "algo-1-zipal",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "model_dir": "/opt/ml/model",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "module_dir": "/opt/ml/code",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "module_name": "train",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "network_interface_name": "eth0",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "num_cpus": 4,
    [36mzuve7o5y6g-algo-1-zipal |[0m     "num_gpus": 0,
    [36mzuve7o5y6g-algo-1-zipal |[0m     "output_data_dir": "/opt/ml/output/data",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "output_dir": "/opt/ml/output",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "output_intermediate_dir": "/opt/ml/output/intermediate",
    [36mzuve7o5y6g-algo-1-zipal |[0m     "resource_config": {
    [36mzuve7o5y6g-algo-1-zipal |[0m         "current_host": "algo-1-zipal",
    [36mzuve7o5y6g-algo-1-zipal |[0m         "hosts": [
    [36mzuve7o5y6g-algo-1-zipal |[0m             "algo-1-zipal"
    [36mzuve7o5y6g-algo-1-zipal |[0m         ]
    [36mzuve7o5y6g-algo-1-zipal |[0m     },
    [36mzuve7o5y6g-algo-1-zipal |[0m     "user_entry_point": "train.py"
    [36mzuve7o5y6g-algo-1-zipal |[0m }
    [36mzuve7o5y6g-algo-1-zipal |[0m 
    [36mzuve7o5y6g-algo-1-zipal |[0m Environment variables:
    [36mzuve7o5y6g-algo-1-zipal |[0m 
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_HOSTS=["algo-1-zipal"]
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_NETWORK_INTERFACE_NAME=eth0
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_HPS={}
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_USER_ENTRY_POINT=train.py
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_FRAMEWORK_PARAMS={}
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_RESOURCE_CONFIG={"current_host":"algo-1-zipal","hosts":["algo-1-zipal"]}
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_INPUT_DATA_CONFIG={"training":{"TrainingInputMode":"File"}}
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_OUTPUT_DATA_DIR=/opt/ml/output/data
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_CHANNELS=["training"]
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_CURRENT_HOST=algo-1-zipal
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_MODULE_NAME=train
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_LOG_LEVEL=20
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_FRAMEWORK_MODULE=
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_INPUT_DIR=/opt/ml/input
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_INPUT_CONFIG_DIR=/opt/ml/input/config
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_OUTPUT_DIR=/opt/ml/output
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_NUM_CPUS=4
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_NUM_GPUS=0
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_MODEL_DIR=/opt/ml/model
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_MODULE_DIR=/opt/ml/code
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"training":"/opt/ml/input/data/training"},"current_host":"algo-1-zipal","framework_module":null,"hosts":["algo-1-zipal"],"hyperparameters":{},"input_config_dir":"/opt/ml/input/config","input_data_config":{"training":{"TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","is_master":true,"job_name":"singleframetrainingtask-2022-12-20-16-37-38-243","log_level":20,"master_hostname":"algo-1-zipal","model_dir":"/opt/ml/model","module_dir":"/opt/ml/code","module_name":"train","network_interface_name":"eth0","num_cpus":4,"num_gpus":0,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_host":"algo-1-zipal","hosts":["algo-1-zipal"]},"user_entry_point":"train.py"}
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_USER_ARGS=[]
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate
    [36mzuve7o5y6g-algo-1-zipal |[0m SM_CHANNEL_TRAINING=/opt/ml/input/data/training
    [36mzuve7o5y6g-algo-1-zipal |[0m PYTHONPATH=/opt/ml/code:/usr/local/bin:/usr/local/lib/python37.zip:/usr/local/lib/python3.7:/usr/local/lib/python3.7/lib-dynload:/usr/local/lib/python3.7/site-packages
    [36mzuve7o5y6g-algo-1-zipal |[0m 
    [36mzuve7o5y6g-algo-1-zipal |[0m Invoking script with the following command:
    [36mzuve7o5y6g-algo-1-zipal |[0m 
    [36mzuve7o5y6g-algo-1-zipal |[0m /usr/local/bin/python train.py
    [36mzuve7o5y6g-algo-1-zipal |[0m 
    [36mzuve7o5y6g-algo-1-zipal |[0m 
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:43.176351 INFO: Welcome to Training Grounds!
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:43.176506 INFO: Loading job
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.377714 INFO: Job of type <class 'SingleFrameTrainingTask__0.tg.common.delivery.sagemaker.job.SagemakerJob'> is loaded
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.377987 INFO: Job has `run` attribute
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.378119 INFO: This is Sagemaker Job performing a training task
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.378330 INFO: Preparing package properties...
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.378547 INFO: {"name": "SingleFrameTrainingTask", "version": "0", "module_name": "SingleFrameTrainingTask__0", "tg_import_path": "SingleFrameTrainingTask__0.tg", "original_tg_import_path": "tg", "resources_location": "/usr/local/lib/python3.7/site-packages/SingleFrameTrainingTask__0/resources"}
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.379271 INFO: Preparing package file...
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.380450 INFO: Processing hyperparameters...
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.380910 INFO: No hyperparameters are provided
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.381000 INFO: Model initialized. Jsonpickling...
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.383134 INFO: Starting training now...
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.474351 INFO: Starting stage 1/1
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.609980 INFO: Saved artifact /opt/ml/model/runs/0/result_df
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.610427 INFO: Saved artifact /opt/ml/model/runs/0/metrics
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.610693 INFO: Saved artifact /opt/ml/model/runs/0/info
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.611373 INFO: Saved artifact /opt/ml/model/runs/0/model
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.611785 INFO: Saved artifact /opt/ml/model/runs/0/training_task
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.612219 INFO: Saved artifact /opt/ml/model/runs/0/train_split
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.612606 INFO: Saved artifact /opt/ml/model/runs/0/test_splits
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.612696 INFO: Completed stage 1/1
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.614647 INFO: ###METRIC###roc_auc_score_test:0.8538095238095237###
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.614752 INFO: ###METRIC###roc_auc_score_train:0.8600247283139194###
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.615094 INFO: Job has exited successfully
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44.615178 INFO: DONE. Exiting Training Grounds.
    [36mzuve7o5y6g-algo-1-zipal |[0m 2022-12-20 16:37:44,919 sagemaker-containers INFO     Reporting training SUCCESS
    [36mzuve7o5y6g-algo-1-zipal exited with code 0
    [0mAborting on container exit...
    ===== Job Complete =====


The result is stored in the local file system in the same format it would be stored in S3. This is a zipped file that contains not only the output, but also the package information:


```python
loader = routine.local().load_result(id)
```

    package.tag.gz
    task.json
    package.json
    runs/
    runs/0/
    runs/0/info.pkl
    runs/0/metrics.pkl
    runs/0/model.pkl
    runs/0/result_df.parquet
    runs/0/test_splits.pkl
    runs/0/train_split.pkl
    runs/0/training_task.pkl


We can now read the dataframe with the results:


```python
df = pd.read_parquet(loader.get_path('runs/0/result_df.parquet'))
```

We can also read pickled objects, although non-directly:


```python
import traceback
try:
    FileIO.read_pickle(loader.get_path('runs/0/training_task.pkl'))
except:
    print(traceback.format_exc())
```

    Traceback (most recent call last):
      File "/tmp/ipykernel_3186/1779605262.py", line 3, in <module>
        FileIO.read_pickle(loader.get_path('runs/0/training_task.pkl'))
      File "/home/yura/anaconda3/envs/fol/lib/python3.8/site-packages/yo_fluq_ds/_misc/io.py", line 17, in read_pickle
        return pickle.load(file)
    ModuleNotFoundError: No module named 'SingleFrameTrainingTask__0'
    


This is due to the fact that the delivered training task was delivered, and the delivery process changed the module name. But the `loader` contains the method to unpickle such files regardless:


```python
loader.unpickle('runs/0/training_task.pkl')
```




    <tg.common.ml.single_frame_training.training_task.SingleFrameTrainingTask at 0x7f5f032d5d00>



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
from tg.common.delivery.sagemaker import Autonamer

Autonamer(build).build_tasks(learning_rate = [1, 2], network_size = [[10], [10, 5]])
```




    [{'info': {'name': 'LR1-NS10'}},
     {'info': {'name': 'LR1-NS10-5'}},
     {'info': {'name': 'LR2-NS10'}},
     {'info': {'name': 'LR2-NS10-5'}}]



As we can see, `Autonamer` will instantiate all the tasks and assign automatically generated names to them. Note that it does not create entry for `context_length` in the name, as it is not variable in this run. 


