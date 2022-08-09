# 3.2. Single frame training (tg.common.ml.single_frame_training)

## Overview of the training process

Training Grounds offers the "training as an object" concept. Instead of "just" writing down the code that is required to train the model, this code is incapsulated in several classes with strong adherence to the SOLID principles. Each class performs the well-defined stage of the process, like splitting dataset into test and train subsets, providing metrics, etc.

The advantages of this approach are:
* Reusability of the most components across the projects, so you don't copy-paste the same code over and over again
* Testability of the components, so you know that you can rely to the components (at least to the part of their functionality that has been tested)

The disadvantage is:
* The onboarding time increases, because "just writing down the code" is a dominating way of teaching data science
* The approach may feel not flexible enough at the start. 

In this model, the training is an object, which is composition of other objects. This object can then be packaged as a Python package, installed in the Docker container and delivered elsewhere, e.g. to Sagemaker. 

## If you don't want all this

... That's also fine. TG is designed to make the life easier, not worse. We offer the SOLID implementation for two wide-spread training scenarios, and we believe that this is a generally better way. But if you are uncomfortable with the SOLID approach to training, or your training process is so specific that it does not fit into both scenarios we have implemented, you always have the following option:

* Inherit from `AbstractTrainingTask`
* Implement `run` method and write the code in any way you see appropriate
* Consider implementing `get_metric_names`, as the metric names must be available prior to the training's start in Sagemaker. You can always return empty array. 

After this, you code will be deliverable with TG delivery. You may also alter the delivery process, as it was explained in the corresponding part of the demo.


## Overview of single-frame training process

Single-frame training process is the process where:
* All data for training fits the memory of the training instance. 
* All training is executed within a single run to `fit` method

These requirements are usually in place for `sklearn` models and alikes. Neural networks require different approach which will be described in the next demo.

## Minimal working example

The single-frame training is represented by `SingleFrameTrainingTask`. To configure it, you may use quite a lot of arguments, so let's go step-by-step and create a minimal working training task.

First, let's load the dataset.


```python
import pandas as pd

df = pd.read_csv('titanic.csv')
df = df.set_index('PassengerId')
for c in ['Pclass','SibSp','Parch','Survived']:
    df[c] = df[c].astype(float)
    
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Let's now define the dataframe with features and labels:


```python
import tg.common.ml.dft as dft

tfac = dft.DataFrameTransformerFactory.default_factory(
    features = [feature for feature in df.columns if feature not in ['Survived','Name','Ticket']],
    max_values_per_category=10
)

tdf = tfac.fit_transform(df)
tdf['Survived'] = df['Survived']
tdf.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Age_missing</th>
      <th>Sex_male</th>
      <th>Sex_female</th>
      <th>Cabin_C23 C25 C27</th>
      <th>Cabin_G6</th>
      <th>...</th>
      <th>Cabin_E101</th>
      <th>Cabin_F33</th>
      <th>Cabin_D</th>
      <th>Cabin_OTHER</th>
      <th>Cabin_NULL</th>
      <th>Embarked_S</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_NULL</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.827377</td>
      <td>-0.530377</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>-0.502445</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.566107</td>
      <td>0.571831</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.786845</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.827377</td>
      <td>-0.254825</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.488854</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.566107</td>
      <td>0.365167</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.420730</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.827377</td>
      <td>0.365167</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.486337</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



Now, we are ready to build and run the very simple training task


```python
import tg.common.ml.single_frame_training as sft

task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider = sft.ModelProvider(sft.ModelConstructor('sklearn.linear_model:LogisticRegression')),
    evaluator = sft.Evaluation.binary_classification
)

result = task.run(tdf)
```


      0%|          | 0/1 [00:00<?, ?it/s]


    2022-08-09 09:26:17.820333+00:00 INFO: Starting stage 1/1
    2022-08-09 09:26:18.007362+00:00 INFO: Completed stage 1/1


Essential components are:

`DataFrameLoader`, which processes pandas data frame into `DataFrameSplit`, containing information about features, labels and splits used for training.

`ModelProvider`, which generates a model. Why do we use a string representation of the class instead of class itself? The reason for that is that the model we use can be a hyperparameter, and this way we can tune the class of the model in the same way as other hyperparameters.

`Evaluation`. This is a function that interprets the output of the model (which is a numpy array) into a dataframe.

As we see, the training produces messages in `Logger`. This is very useful when monitoring the training in Sagemaker, however, does not help much in this notebook, so we will disable `Logger`.


```python
from tg.common import Logger

Logger.disable()
```

The result is a dictionary that contains all information about the training, for intance, model itself:


```python
result['runs'][0]['model']
```




    Pipeline(steps=[('ColumnNamesKeeper', ColumnNamesKeeper()),
                    ('Model', LogisticRegression())])



Note that the model is a pipeline of two steps.

The first step of the pipeline is a decorator that keeps the column names of the original dataset. It is mostly for double-checking, so when you use the model in production you can check if all the columns are present. You can disable adding this with a parameter of `ModelProvider`, but we recommend to let it be.

The second step is a model, specified by a `ModelProvider`.

Also, the result contains the dataframe, created by `Evaluation`. This table is useful to compute various metrics:


```python
result['runs'][0]['result_df'].head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted</th>
      <th>true</th>
      <th>stage</th>
      <th>original_index</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.096888</td>
      <td>0.0</td>
      <td>train</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.938098</td>
      <td>1.0</td>
      <td>train</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.649651</td>
      <td>1.0</td>
      <td>train</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.916243</td>
      <td>1.0</td>
      <td>train</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.080957</td>
      <td>0.0</td>
      <td>train</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## Spliters

In the previous step we trained model on all the available data, and then applied the model to this training data, without having a test set. To change that, we need to use splitter.


```python
import tg.common.ml.single_frame_training as sft

task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider = sft.ModelProvider(sft.ModelConstructor('sklearn.linear_model:LogisticRegression')),
    evaluator = sft.Evaluation.binary_classification,
    splitter = sft.FoldSplitter(fold_count=5)
)

result = task.run(tdf)
```


      0%|          | 0/5 [00:00<?, ?it/s]


Now, the result will have 5 runs (one run for each split). Let's check that we have some tests in each run, and that the rows are different for different runs


```python
for run in result['runs'].values():
    rdf = run['result_df']
    rdf = rdf.loc[rdf.stage=='test']
    print(rdf.shape[0], list(rdf.sort_index().index[:5]))
```

    268 [2, 3, 6, 9, 11]
    268 [1, 3, 4, 7, 9]
    268 [2, 3, 5, 8, 11]
    268 [4, 5, 6, 11, 15]
    268 [12, 14, 18, 34, 35]


We implemented basic K-fold split. By using other parameters of `FoldSplitter`, or combining several `FoldSplitter` by `UnionSplitter` and `CompositionSplitter`, you can achive more complicated configurations with e.g. holdout test set. Proper splitting of the dataset is extremely important to prevent data leak, so we suggest you will think the split architecture through, and maybe even add your own splitters.

For instance, when analyzing time-dependent data, we find `TimeSplitter` to extremely useful. This splitter forms splits by time. Assume $T_0$ is the time when our data start, $\Delta$ is a step (i.e. one month). In this case, `TimeSplitter` will create the following splits:

* train on $[T_0, T_0+\Delta]$, test on $[T_0+\Delta, T_0+2\Delta]$
* train on $[T_0, T_0+2\Delta]$, test on $[T_0+2\Delta, T_0+3\Delta]$
* train on $[T_0, T_0+3\Delta]$, test on $[T_0+3\Delta, T_0+4\Delta]$
* etc



To uncover a bit what the split is actually doing, let's run it manually:


```python
initial_split = sft.DataFrameLoader('Survived').get_data(tdf)
splits = sft.FoldSplitter(5)(initial_split)
split_repr = splits[0].__dict__
del split_repr['df']
split_repr
```




    {'features': ['Pclass',
      'Age',
      'SibSp',
      'Parch',
      'Fare',
      'Age_missing',
      'Sex_male',
      'Sex_female',
      'Cabin_C23 C25 C27',
      'Cabin_G6',
      'Cabin_B96 B98',
      'Cabin_F2',
      'Cabin_C22 C26',
      'Cabin_E101',
      'Cabin_F33',
      'Cabin_D',
      'Cabin_OTHER',
      'Cabin_NULL',
      'Embarked_S',
      'Embarked_C',
      'Embarked_Q',
      'Embarked_NULL'],
     'labels': 'Survived',
     'train': Int64Index([  1,   4,   5,   7,   8,  10,  12,  13,  14,  16,
                 ...
                 874, 875, 877, 880, 882, 885, 886, 887, 889, 890],
                dtype='int64', name='PassengerId', length=623),
     'tests': {'test': Int64Index([  2,   3,   6,   9,  11,  15,  19,  21,  28,  31,
                  ...
                  849, 853, 876, 878, 879, 881, 883, 884, 888, 891],
                 dtype='int64', name='PassengerId', length=268)},
     'info': {'test': {'fold': 0, 'index': 0, 'split_column': 'index'}}}



The `DataFrameSplit` object has and `features`, `labels` fields inherited from the `initial_split`. It also inherits `df`, but we have removed it from output for readability. `train` is the array of indices for which train is performed, and `tests` is a dictionary of indices belonging to the one or more test sets (e.g. validation and holdout). Essentially, all splitters form these two fields, `train` and `tests`.

## Transformers



The previous example is actually very flawed. Note that transformer is applied to _all_ the data, while it has to be applied to the training data only. So there is a data leak in this example via the transformer, and this should not happen. This is why we **strongly** encourage you to make transformer a part of the model:


```python
import tg.common.ml.single_frame_training as sft

task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider = sft.ModelProvider(
        constructor = sft.ModelConstructor('sklearn.linear_model:LogisticRegression'),
        transformer = tfac),
    evaluator = sft.Evaluation.binary_classification,
    splitter = sft.FoldSplitter(fold_count=5)
)

result = task.run(df)
```


      0%|          | 0/5 [00:00<?, ?it/s]


If we now look at the result:


```python
result['runs'][0]['model']
```




    Pipeline(steps=[('ColumnNamesKeeper', ColumnNamesKeeper()),
                    ('Transformer',
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7f9d34046370>),
                    ('ColumnNamesKeeperAfterTransformation', ColumnNamesKeeper()),
                    ('Model', LogisticRegression())])



We see that now 2 more steps are added to the pipeline. The first is our `DataFrameTransformerFactory`, which acts as a transformer. The second is yet another instance of ColumnNamesKeeper(), which allows us to track the column names even after transformation


```python
print(result['runs'][0]['model'][0].column_names_)
print(result['runs'][0]['model'][2].column_names_)
```

    ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Age_missing', 'Sex_male', 'Sex_female', 'Cabin_C23 C25 C27', 'Cabin_E101', 'Cabin_B96 B98', 'Cabin_C22 C26', 'Cabin_D', 'Cabin_C93', 'Cabin_C123', 'Cabin_D33', 'Cabin_OTHER', 'Cabin_NULL', 'Embarked_S', 'Embarked_C', 'Embarked_Q', 'Embarked_NULL']


## Metrics

You can use `MetricsPool` to measure the training success:


```python
import tg.common.ml.single_frame_training as sft
from sklearn.metrics import roc_auc_score

task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider = sft.ModelProvider(
        constructor = sft.ModelConstructor('sklearn.linear_model:LogisticRegression'),
        transformer = tfac),
    evaluator = sft.Evaluation.binary_classification,
    splitter = sft.FoldSplitter(fold_count=5),
    metrics_pool = sft.MetricPool().add_sklearn(roc_auc_score)
)

result = task.run(df)
```


      0%|          | 0/5 [00:00<?, ?it/s]



```python
from yo_fluq_ds import Query

pd.DataFrame([run['metrics'] for run in result['runs'].values()])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roc_auc_score_test</th>
      <th>roc_auc_score_train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.856012</td>
      <td>0.863213</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.825860</td>
      <td>0.877670</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.831626</td>
      <td>0.875503</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.823640</td>
      <td>0.877365</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.876280</td>
      <td>0.855036</td>
    </tr>
  </tbody>
</table>
</div>



Note: since `sft.Evaluation.binary_classification` actually uses `predict_proba` instead of `predict`, f1, precision and recall scores cannot be used with it. The best way around it is using custom metric. Note that you can combine several metrics into one `Metric` instance:


```python
import tg.common.ml.single_frame_training as sft
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

class MetricsWithBorderline(sft.Metric):
    def __init__(self, borderline=0.5):
        self.borderline = borderline
    def get_names(self):
        return ["f1","precision","recall"]
    def measure(self, result_df: pd.DataFrame, source_data):
        prediction = result_df.predicted>self.borderline
        return [
            f1_score(result_df.true, prediction),
            precision_score(result_df.true, prediction),
            recall_score(result_df.true, prediction)
        ]
            
    

task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider = sft.ModelProvider(
        constructor = sft.ModelConstructor('sklearn.linear_model:LogisticRegression'),
        transformer = tfac),
    evaluator = sft.Evaluation.binary_classification,
    splitter = sft.FoldSplitter(fold_count=5),
    metrics_pool = sft.MetricPool().add_sklearn(roc_auc_score).add(MetricsWithBorderline()),
)

result = task.run(df)

pd.DataFrame([run['metrics'] for run in result['runs'].values()])
```


      0%|          | 0/5 [00:00<?, ?it/s]





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roc_auc_score_test</th>
      <th>f1_test</th>
      <th>precision_test</th>
      <th>recall_test</th>
      <th>roc_auc_score_train</th>
      <th>f1_train</th>
      <th>precision_train</th>
      <th>recall_train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.856012</td>
      <td>0.734694</td>
      <td>0.750000</td>
      <td>0.720000</td>
      <td>0.863213</td>
      <td>0.736383</td>
      <td>0.778802</td>
      <td>0.698347</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.825860</td>
      <td>0.746544</td>
      <td>0.794118</td>
      <td>0.704348</td>
      <td>0.877670</td>
      <td>0.736364</td>
      <td>0.760563</td>
      <td>0.713656</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.831626</td>
      <td>0.706468</td>
      <td>0.763441</td>
      <td>0.657407</td>
      <td>0.875503</td>
      <td>0.738938</td>
      <td>0.766055</td>
      <td>0.713675</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.823640</td>
      <td>0.720379</td>
      <td>0.710280</td>
      <td>0.730769</td>
      <td>0.877365</td>
      <td>0.747253</td>
      <td>0.783410</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.876280</td>
      <td>0.727273</td>
      <td>0.744186</td>
      <td>0.711111</td>
      <td>0.855036</td>
      <td>0.738776</td>
      <td>0.760504</td>
      <td>0.718254</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, `Metric::measure` accepts not only resulting dataframe (which is essentialy output of the evaluator), but also the source data used for training, so you can weight different instances based on their importance, etc.

## Using other classifiers

Needless to say, the whole system works not only with logistic regression, but also with any other `sklearn` algorithm, such as XGBoost or random forests


```python
import tg.common.ml.single_frame_training as sft
from sklearn.metrics import roc_auc_score, f1_score

task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider = sft.ModelProvider(
        constructor = sft.ModelConstructor('sklearn.ensemble:RandomForestClassifier'),
        transformer = tfac),
    evaluator = sft.Evaluation.binary_classification,
    splitter = sft.FoldSplitter(fold_count=5),
    metrics_pool = sft.MetricPool().add_sklearn(roc_auc_score),
)

result = task.run(df)

pd.DataFrame([run['metrics'] for run in result['runs'].values()])
```


      0%|          | 0/5 [00:00<?, ?it/s]





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roc_auc_score_test</th>
      <th>roc_auc_score_train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.865179</td>
      <td>0.998308</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.840779</td>
      <td>0.998793</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.846325</td>
      <td>0.997314</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.826601</td>
      <td>0.999662</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.877310</td>
      <td>0.997893</td>
    </tr>
  </tbody>
</table>
</div>



## Using catboost

`catboost` is a powerful algorithm, based on decision tree, which works particularly well on datasets with lots of categorical variables. We admit that the true reason why we pay specific attention to this algorithm is because the major contributor to TG is Russian, and the `catboost` was also developed in Russia, and widespread in the data science circles there. 

There are two moments about this algorithm.

First, `catboost` doesn't require you to apply OneHot encoding, so we will exclude this step from `tfac`. It will also reduce the memory, required for the training. However, **be careful with that**! If you don't trim the amount of variables in your categorical columns, there are the great chances that `catboost` will basically build a binary tree of exponential size around these columns, which will make the model huge and slow! `TopKPopularStrategy` actually appeared in code after we discovered this sad fact.

Second, when using `catboost`, there is a little trick to be make. The list of categorical variables must be given to catboost prior training. However, in our architecture, we don't really know which columns are categorical, because it is only known when TransformerFactory processed the dataset. This is why we use a little wrapper to do that.


```python
import tg.common.ml.single_frame_training as sft
from sklearn.metrics import roc_auc_score, f1_score
from functools import partial

catboost_tfac = (dft.DataFrameTransformerFactory()
 .with_feature_block_list(['Survived','Name','Ticket'])
 .on_continuous(dft.ContinousTransformer)
 .on_categorical(dft.CategoricalTransformer)
 .on_rich_category(10, partial(
     dft.CategoricalTransformer, 
     replacement_strategy = dft.TopKPopularStrategy(10,'OTHER')
)))


task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider = sft.ModelProvider(
        constructor = sft.ModelConstructor('catboost:CatBoostClassifier', silent=True),
        transformer = catboost_tfac,
        model_fix = sft.ModelProvider.catboost_model_fix
    ),
    evaluator = sft.Evaluation.binary_classification,
    splitter = sft.FoldSplitter(fold_count=5),
    metrics_pool = sft.MetricPool().add_sklearn(roc_auc_score),
)

result = task.run(df)

pd.DataFrame([run['metrics'] for run in result['runs'].values()])
```


      0%|          | 0/5 [00:00<?, ?it/s]





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roc_auc_score_test</th>
      <th>roc_auc_score_train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.880298</td>
      <td>0.931954</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.832793</td>
      <td>0.935639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.853791</td>
      <td>0.942115</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.839030</td>
      <td>0.941362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.890886</td>
      <td>0.921031</td>
    </tr>
  </tbody>
</table>
</div>



Note that we provided some additional arguments to the `ModelConstructor`. This parameter is passed to `CatBoostClassifier` constructor, making it silent. The same works with other models, too.

Now, let's check the trained model:


```python
result['runs'][0]['model']
```




    Pipeline(steps=[('ColumnNamesKeeper', ColumnNamesKeeper()),
                    ('Transformer',
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7f9d350c1460>),
                    ('ColumnNamesKeeperAfterTransformation', ColumnNamesKeeper()),
                    ('Model',
                     Pipeline(steps=[('CategoricalVariablesSetter',
                                      <tg.common.ml.single_frame_training.model_provider.CatBoostWrap object at 0x7f9d2b5b2670>),
                                     ('Model',
                                      <catboost.core.CatBoostClassifier object at 0x7f9d350c12b0>)]))])



`model_fix` is a function, that updates the model to something else. In our case, the initial instance of catboost model `Model` was replaced with a Pipeline, containing two steps. The second step is `Model`. The first step is a wrapper, that accepts the dataset, processed by transformers, understands which columns are categorical, and then sets the list of this columns to the `Model`

## Artificiers

_Artificier_ is an interface to inject an arbitrary code to the training process. So far, we have the following use cases for artificiers:
* Remove model from the training result. The model may be huge and we may not be even interested in the model per se, just by it's metrics.
* Get the feature significance. Many algorithms allow us to extract feature significance from the model, which can be used in business analysis without the model itself.
* Augment the resulting df with additional columns to compute metrics better

Let's use write an artificier to discover the most important features in our dataset. 


```python
import tg.common.ml.single_frame_training as sft
from sklearn.metrics import roc_auc_score

class SignificanceArtificier(sft.Artificier):
    def run_before_storage(self, model_info):
        column_names_keeper = model_info.result.model[2] # type: sft.ColumnNamesKeeper
        column_names = column_names_keeper.column_names_
        coeficients = model_info.result.model[3].coef_
        model_info.result.significance = pd.Series(coeficients[0], index=column_names)
        

task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider = sft.ModelProvider(
        constructor = sft.ModelConstructor('sklearn.linear_model:LogisticRegression'),
        transformer = tfac),
    evaluator = sft.Evaluation.binary_classification,
    splitter = sft.FoldSplitter(fold_count=50),
    metrics_pool = sft.MetricPool().add_sklearn(roc_auc_score),
    artificers=[
        SignificanceArtificier()
    ]
)

result = task.run(df)
```


      0%|          | 0/50 [00:00<?, ?it/s]


**Note**: `Artificier` has two methods:

* `run_before_metrics` should be used, if the artificier computes something that is required by metrics
* `run_before_storage` should be used in other cases.

Both methods are implemented in `Artificier` base class as stubs, so you only need to define the one you are going to use.

Let's browse the result of our `SignificanceArtificer`.


```python
sdf = pd.DataFrame([run['significance'] for run in result['runs'].values()])
sdf.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Age_missing</th>
      <th>Sex_male</th>
      <th>Sex_female</th>
      <th>Cabin_C23 C25 C27</th>
      <th>Cabin_E101</th>
      <th>...</th>
      <th>Cabin_B77</th>
      <th>Cabin_E44</th>
      <th>Cabin_C2</th>
      <th>Cabin_D20</th>
      <th>Cabin_B35</th>
      <th>Cabin_F G73</th>
      <th>Cabin_B20</th>
      <th>Cabin_B18</th>
      <th>Cabin_B22</th>
      <th>Cabin_C52</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.729712</td>
      <td>-0.599334</td>
      <td>-0.442507</td>
      <td>-0.090442</td>
      <td>0.044117</td>
      <td>-0.275676</td>
      <td>-1.335706</td>
      <td>1.334984</td>
      <td>0.096968</td>
      <td>0.280976</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.827425</td>
      <td>-0.629947</td>
      <td>-0.350360</td>
      <td>0.028300</td>
      <td>-0.068819</td>
      <td>-0.388938</td>
      <td>-1.391383</td>
      <td>1.391777</td>
      <td>-0.203659</td>
      <td>0.239313</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.827155</td>
      <td>-0.694813</td>
      <td>-0.397645</td>
      <td>-0.046530</td>
      <td>-0.041126</td>
      <td>-0.330843</td>
      <td>-1.408049</td>
      <td>1.410029</td>
      <td>-0.007618</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.730623</td>
      <td>-0.655058</td>
      <td>-0.473254</td>
      <td>-0.064541</td>
      <td>0.241218</td>
      <td>-0.183655</td>
      <td>-1.388633</td>
      <td>1.388427</td>
      <td>-0.074086</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.747540</td>
      <td>-0.535634</td>
      <td>-0.278574</td>
      <td>-0.010801</td>
      <td>0.099159</td>
      <td>-0.465550</td>
      <td>-1.235044</td>
      <td>1.235681</td>
      <td>-0.270592</td>
      <td>0.421581</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 57 columns</p>
</div>



Apparently the most popular cabins depend strongly on the split, so we will remove them. After that, we may draw a violinplot demonstrating significance.


```python
from seaborn import violinplot
from matplotlib import pyplot as plt

bad_columns = sdf.isnull().any(axis=0)
bad_columns = list(bad_columns.loc[bad_columns].index)
cdf = sdf[[c for c in sdf.columns if c not in bad_columns]]

_, ax = plt.subplots(1,1,figsize=(20,5))
violinplot(data=cdf, ax=ax)
pass
```


    
![png](README_images/tg.common.ml.single_frame_training_output_47_0.png?raw=true)
    


## Hyperparameter optimization

Hyperparameter optimization can be performed over *any* field inside the task class. 


```python
task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider = sft.ModelProvider(
        constructor = sft.ModelConstructor('sklearn.linear_model:LogisticRegression'),
        transformer = tfac),
    evaluator = sft.Evaluation.binary_classification,
    splitter = sft.FoldSplitter(fold_count=1),
    metrics_pool = sft.MetricPool().add_sklearn(roc_auc_score)
)
parameters = {
    "model_provider.constructor.kwargs.C" : 1,
    "model_provider.constructor.kwargs.penalty": 'l2',
}
task.apply_hyperparams(parameters)

result = task.run(df)
result['runs'][0]['model']
```


      0%|          | 0/1 [00:00<?, ?it/s]





    Pipeline(steps=[('ColumnNamesKeeper', ColumnNamesKeeper()),
                    ('Transformer',
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7f9d1d251700>),
                    ('ColumnNamesKeeperAfterTransformation', ColumnNamesKeeper()),
                    ('Model', LogisticRegression(C=1))])



When working with sagemaker, the hyperparameters are passed to the model in string form. So, you will need to indicate type as well:


```python
task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider = sft.ModelProvider(
        constructor = sft.ModelConstructor('sklearn.linear_model:LogisticRegression'),
        transformer = tfac),
    evaluator = sft.Evaluation.binary_classification,
    splitter = sft.FoldSplitter(fold_count=1),
    metrics_pool = sft.MetricPool().add_sklearn(roc_auc_score)
)
parameters = {
    "model_provider.constructor.kwargs.C:float" : '1',
    "model_provider.constructor.kwargs.penalty": 'l2',
}
task.apply_hyperparams(parameters)

result = task.run(df)
result['runs'][0]['model']
```


      0%|          | 0/1 [00:00<?, ?it/s]





    Pipeline(steps=[('ColumnNamesKeeper', ColumnNamesKeeper()),
                    ('Transformer',
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7f9d1dec8130>),
                    ('ColumnNamesKeeperAfterTransformation', ColumnNamesKeeper()),
                    ('Model', LogisticRegression())])



With Training Grounds, it is possible to perform hyperparameter optimization of single-frame model. If the model requires a significant time to train, we should use sagemaker and hyperopt. But sometimes it can be executed locally. For that, we offer `Kraken` class. 

Kraken does exactly one thing: it executes any given method over set of parameters, and brings the result into big pandas dataframe. Kraken supports exception handling as well as caching intermediate result on the disk for further restart, and this functionality is well-tested.


```python
from tg.common.ml.miscellaneous import Kraken

config = [
    {'a': 1, 'b': 2},
    {'a': 3, 'b': 4}
]

def method(iteration, a,b):
    return pd.DataFrame([dict(c=a+b)])

Kraken.release(method,config)
```


      0%|          | 0/2 [00:00<?, ?it/s]





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c</th>
      <th>iteration</th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



So, single frame task can be used with Kraken for parameter optimization. 


```python
task_configs = [
    {"model_provider.constructor.type_name" : "sklearn.linear_model:LogisticRegression"},
    {"model_provider.constructor.type_name" : "sklearn.ensemble:RandomForestClassifier"},
]

task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider = sft.ModelProvider(
        constructor = sft.ModelConstructor('sklearn.linear_model:LogisticRegression'),
        transformer = tfac),
    evaluator = sft.Evaluation.binary_classification,
    splitter = sft.FoldSplitter(fold_count=20),
    metrics_pool = sft.MetricPool().add_sklearn(roc_auc_score)
)

method, configs = task.make_kraken_task(task_configs, df)
rdf = Kraken.release(method,configs,lambda z, _: pd.DataFrame([z.metrics]))

```


      0%|          | 0/40 [00:00<?, ?it/s]



```python
from matplotlib import pyplot as plt
_, ax = plt.subplots(2,1,figsize=(20,10))
violinplot(data=rdf, 
           x='roc_auc_score_test', 
           y='hyperparameters_model_provider.constructor.type_name', 
           orient='horisontal',
           ax=ax[0]
          )
violinplot(data=rdf, 
           x='roc_auc_score_train', 
           y='hyperparameters_model_provider.constructor.type_name', 
           orient='horisontal',
           ax=ax[1])
pass
```


    
![png](README_images/tg.common.ml.single_frame_training_output_56_0.png?raw=true)
    


We see the classic image for logistic regression  and random forest: while they perform comparable on the test set, the random forests fit extremely well on the train set.
