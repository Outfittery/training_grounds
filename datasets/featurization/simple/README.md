# 2.3.1. Simple Featurization Jobs and Datasets (tg.common.datasets.featurization.simple)


`FeaturizationJob` is the job that combines `DataSource` with `Selector`, addressing the production-level questions of memory control and results output. 

To demonstrate how it works, we will first create data source and selector in a way similar to previous demo, but without artificial distortion.


```python
from tg.common.datasets.access import MockDfDataSource
import pandas as pd

source = MockDfDataSource(pd.read_csv('titanic.csv'))
selector = lambda z: z
selector(source.get_data().first())
```




    {'PassengerId': 1, 'Survived': 0, 'Pclass': 3, 'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male', 'Age': 22.0, 'SibSp': 1, 'Parch': 0, 'Ticket': 'A/5 21171', 'Fare': 7.25, 'Cabin': nan, 'Embarked': 'S'}



`MockDfDataSource` is a class that converts the dataframe into DOF of its rows. You may effectively use this class for, e.g., unit tests. And since the output of the `MockDfDataSource` is already in an appropriate format, we don't need any complex selectors, so we will just use an identity function.

Now we can create a data frame in the most primitive way:


```python
df = pd.DataFrame(selector(z) for z in source.get_data())
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



The featurization job that produces the same result can be created as follows:


```python
from tg.common.datasets.featurization import FeaturizationJob, DataframeFeaturizer
from tg.common import MemoryFileSyncer

mem = MemoryFileSyncer()

job = FeaturizationJob(
    name = 'job',
    version = 'v1',
    source = source,
    featurizers = {
        'passengers': DataframeFeaturizer(row_selector = selector)
    },
    syncer = mem,
    location = './temp/featurization_job'
)

job.run()
```

    2022-08-09 09:25:59.954577+00:00 INFO: Featurization Job job at version v1 has started
    2022-08-09 09:25:59.956774+00:00 INFO: Fetching data
    2022-08-09 09:26:00.076865+00:00 INFO: Data fetched, finalizing
    2022-08-09 09:26:00.197256+00:00 INFO: Uploading data
    2022-08-09 09:26:00.200872+00:00 INFO: Featurization job completed


Some notes: 

* `DataFrameFeaturizer`: When used in this way, it just applies `row_selector` to each data object from `source` and collects the results into pandas dataframes
* If no `location` is provided, the folder will be created automatically in the `Loc.temp_path` folder. Usually we don't care where the intermediate files are stored, as syncer takes care of them automatically.
* `MemoryFileSyncer`. The job creates files locally (in the `location` folder), and the uploads them to the remote destination. For demonstration purposes, we will "upload" data in the memory. `tg.common` also contains `S3FileSyncer` that syncs the files with `S3`. Interfaces for other storages may be written, deriving from `FileSyncer`. Essentialy, the meaning of `FileSyncer` is a connection between a specific location on the local disk and the location somewhere else. When calling `upload` or `download` methods, the class assures the same content of given files/folders.


The resulting files can be viewed in the following way:


```python
list(mem.cache)
```




    ['passengers/f4b88092-ab42-43f8-baa1-f312e7777e65.parquet']



Method `get_parquet`, used for testing purposes, will read the file with the given key as a parquet file. Instead of the file name, the index of this name in the list of files can be provided, or a lambda-expression that filters the name you want to read.


```python
mem.get_parquet(0).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>None</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## Partitioning

What if the data is too big? Per se, it's not a problem: data sources do not normally keep all the data in memory at once, and selectors process the data one-by-one. But when we do the last step of assembling the data into a dataframe, we might run into problem. Let's use `DataframeFeaturizer` arguments to prevent it:


```python
mem = MemoryFileSyncer()

job = FeaturizationJob(
    name = 'job',
    version = 'v1',
    source = source,
    featurizers = {
        'passengers': DataframeFeaturizer(buffer_size=250, row_selector = selector)
    },
    syncer = mem
)

job.run()
list(mem.cache)
```

    2022-08-09 09:26:00.309620+00:00 INFO: Featurization Job job at version v1 has started
    2022-08-09 09:26:00.313845+00:00 INFO: Fetching data
    2022-08-09 09:26:00.438774+00:00 INFO: Data fetched, finalizing
    2022-08-09 09:26:00.445772+00:00 INFO: Uploading data
    2022-08-09 09:26:00.447250+00:00 INFO: Featurization job completed





    ['passengers/58e480ec-a8f6-41b9-a0f7-50620b0904ea.parquet',
     'passengers/069141c4-7ef1-420f-90ef-1a409fa0f01d.parquet',
     'passengers/fca730fd-e50e-468b-aaca-dd53cf548296.parquet',
     'passengers/15f4f91b-4f1d-4a22-b072-dab7c12dc25f.parquet']




```python
len(mem.get_parquet(0))
```




    250



Here we have limited amount of rows that can be put into one data frame to 250. As the result, each data frame in `memory.cache` has no more than 250 rows, and we have several files in our dataset.

## Filtering / expanding

What if our data are more complicated, and there is no 1-to-1 correspondance between data objects and rows? Examples are:
* We want to filter out some rows. In this case, 1 incoming data object corresponds to 0 rows.
* We are processing data that are organized not as a flow of passengers, but as a flow of cabins, where each cabin is a list of passengers. In this case, 1 incoming data object corresponds to arbitrary amount of rows.

Let's implement the first option by modifying `DataFrameFeaturizer`, and also explore some additional features of this class


```python
import numpy as np

class MyDataFrameFeaturizer(DataframeFeaturizer):
    def __init__(self):
        super(MyDataFrameFeaturizer, self).__init__()
        
    def _featurize(self, obj):
        if obj['Age'] < 30:
            return []
        else:
            return [obj]
        
    def _postprocess(self, df):
        df.Cabin = np.where(df.Cabin.isnull(), 'NONE', df.Cabin)
        return df
     
mem = MemoryFileSyncer()

job = FeaturizationJob(
    name = 'job',
    version = 'v1',
    source = source,
    featurizers = {
        'passengers': MyDataFrameFeaturizer()
    },
    syncer = mem
)

job.run()
mem.get_parquet(0).sort_values('Age').head()
```

    2022-08-09 09:26:00.474908+00:00 INFO: Featurization Job job at version v1 has started
    2022-08-09 09:26:00.478526+00:00 INFO: Fetching data
    2022-08-09 09:26:00.570513+00:00 INFO: Data fetched, finalizing
    2022-08-09 09:26:00.579881+00:00 INFO: Uploading data
    2022-08-09 09:26:00.580905+00:00 INFO: Featurization job completed





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
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
  </thead>
  <tbody>
    <tr>
      <th>349</th>
      <td>607</td>
      <td>0</td>
      <td>3</td>
      <td>Karaic, Mr. Milan</td>
      <td>male</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
      <td>349246</td>
      <td>7.8958</td>
      <td>NONE</td>
      <td>S</td>
    </tr>
    <tr>
      <th>88</th>
      <td>179</td>
      <td>0</td>
      <td>2</td>
      <td>Hale, Mr. Reginald</td>
      <td>male</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
      <td>250653</td>
      <td>13.0000</td>
      <td>NONE</td>
      <td>S</td>
    </tr>
    <tr>
      <th>76</th>
      <td>158</td>
      <td>0</td>
      <td>3</td>
      <td>Corn, Mr. Harry</td>
      <td>male</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392090</td>
      <td>8.0500</td>
      <td>NONE</td>
      <td>S</td>
    </tr>
    <tr>
      <th>287</th>
      <td>521</td>
      <td>1</td>
      <td>1</td>
      <td>Perreault, Miss. Anne</td>
      <td>female</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
      <td>12749</td>
      <td>93.5000</td>
      <td>B73</td>
      <td>S</td>
    </tr>
    <tr>
      <th>156</th>
      <td>287</td>
      <td>1</td>
      <td>3</td>
      <td>de Mulder, Mr. Theodore</td>
      <td>male</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
      <td>345774</td>
      <td>9.5000</td>
      <td>NONE</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Here we have created a special class just for this particular dataset, therefore, we don't really need to pass the `selector` to it.

In `_featurize` method we process a given data object in an arbitrary way, and return a list of rows.

In `_postprocess` we may perform some additional operations on the dataframe. We have imputed the values for `Cabin` field, but **do not do it** in the real examples: the imputation belongs to the machine learning part of the pipeline, not to the data cleaning.

## Aggregating 

Sometimes we are not really interested in the dataframe as it is, but want to compute some aggregated statistics and use it as features. In our case, we may wish to compute average fare and age for each cabin. (_Note that it would be an awful idea in the reality, as it would be a leakage of data from test to train_).

In this case, we need to step back in our inheritance hierarchy, and use `StreamFeaturizer` class.


```python
from tg.common.datasets.featurization import StreamFeaturizer

class CabinStatisticsFeaturizer(StreamFeaturizer):
    def start(self):
        self.cabins = {}
    
    def observe_data_point(self, item):
        cabin =  item['Cabin'] 
        if not isinstance(cabin, str) or item['Fare'] is None or item['Age'] is None:
            return
        if cabin not in self.cabins:
            self.cabins[cabin] = dict(count=0, age=0, fare=0, id=cabin)
        self.cabins[cabin]['count']+=1
        self.cabins[cabin]['age']+=item['Age']
        self.cabins[cabin]['fare']+=item['Fare']
        
    def finish(self):
        df = pd.DataFrame(list(self.cabins.values()))
        df.age = df.age/df['count']
        df.fare = df.fare/df['count']
        return df.set_index('id')
        
dataset_buffer = MemoryFileSyncer()

job = FeaturizationJob(
    name = 'job',
    version = 'v1',
    source = source,
    featurizers = {
        'passengers': MyDataFrameFeaturizer(),
        'cabins': CabinStatisticsFeaturizer()
    },
    syncer = dataset_buffer,
    location='./temp/test'
)

job.run()
list(dataset_buffer.cache)
```

    2022-08-09 09:26:00.609982+00:00 INFO: Featurization Job job at version v1 has started
    2022-08-09 09:26:00.611226+00:00 INFO: Fetching data
    2022-08-09 09:26:00.701630+00:00 INFO: Data fetched, finalizing
    2022-08-09 09:26:00.714961+00:00 INFO: Uploading data
    2022-08-09 09:26:00.716187+00:00 INFO: Featurization job completed





    ['cabins/5ae20540-a3ff-44e2-b5c2-c2da7548560f.parquet',
     'passengers/d2244032-3003-437d-8de1-ccac93cd2c42.parquet']



These are features for cabins. Note, that passengers' features are produced as well.


```python
dataset_buffer.get_parquet(lambda z: z.startswith('cabins')).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>age</th>
      <th>fare</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C85</th>
      <td>1</td>
      <td>38.00</td>
      <td>71.28330</td>
    </tr>
    <tr>
      <th>C123</th>
      <td>2</td>
      <td>36.00</td>
      <td>53.10000</td>
    </tr>
    <tr>
      <th>E46</th>
      <td>1</td>
      <td>54.00</td>
      <td>51.86250</td>
    </tr>
    <tr>
      <th>G6</th>
      <td>4</td>
      <td>14.75</td>
      <td>13.58125</td>
    </tr>
    <tr>
      <th>C103</th>
      <td>1</td>
      <td>58.00</td>
      <td>26.55000</td>
    </tr>
  </tbody>
</table>
</div>



So it is possible, and often very useful, to build several datasets with a single run over the source. So far, retrieving data from source was the most time-consuming part of the featurization, so it _really_ saves time

## Datasets

The result of the FeaturizerJob can be easily consumed with the `Dataset` class. The `Dataset` class downloads the data from the remote location with `FileSyncer`, and allows you to open the files.

If the `FeaturizationJob` creates several folders, as in the example above, **one** of them is a `Dataset`. If you need access to both of them, you need to create two instances of `Dataset` class.

For demonstration, we will use `dataset_buffer` created at the last demonstration of the `FeaturizationJob`, with two datasets, `cabins` and `passengers`.


```python
from tg.common.datasets.featurization import Dataset

dataset = Dataset(
    './temp/dataset',
    dataset_buffer.change_remote_subfolder('passengers')
)
```

As we remember, `FileSyncer` establishes connection between local drive and the remote datasource. Let's dwell into some technicalities here. Method `change_remote_subfolder`:
* returns a new instance of `FileSyncer`, because this class is immutable by design
* establishes connection between **the same** folder at the local drive, **changing only subfolder** on the remote storage. 

This is because in this particular case local folder does not matter, as it will be overriden in `Dataset` constructor. In fact, it's not even set:


```python
dataset_buffer.get_local_folder() is None
```




    True



To set the local folder, use `change_local_folder` method. We use the word _folder_ for local drive, because it can be anywhere. We use _subfolder_ for the remote source, as the path can only direct to a subfolder of the initially-defined folder. If you use `S3FileSyncer` and set it to a specific bucket with a specific prefix, you won't be able to escape from this prefix, which increases safety of operations.

If you want to change both simultaneously, spawning `FileSyncer` instance fo a subfolder in both local drive and remote storage, use `cd` method.

Back to the datasets. First, you need to download dataset:


```python
dataset.download()
dataset.read().head(3)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NONE</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Datasets can be very large. This problem is avoided on the stage of dataset's creation by partitioning, but may appear again if you try to read the entire dataset in the memory. The following procedure of dataset's discovery is recommended.

First, read a bit of rows from the dataset. In case of partitioned dataset, only some partition will be opened:


```python
dataset.read(count=3)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NONE</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Often, you only need a specific columns from dataset. You may explore the columns names by reading several rows, and then specify columns you need:


```python
dataset.read(columns=['Age','Sex','Survived']).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38.0</td>
      <td>female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.0</td>
      <td>female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.0</td>
      <td>male</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>male</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54.0</td>
      <td>male</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



If you are only interested in some rows from dataset, use `selector` argument. It will allow you to `loc` dataframe on the records you need. It will be done, again, per partition, so the memory won't overfill.


```python
dataset.read(selector=lambda z: z.loc[z.Age==80])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
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
  </thead>
  <tbody>
    <tr>
      <th>361</th>
      <td>631</td>
      <td>1</td>
      <td>1</td>
      <td>Barkworth, Mr. Algernon Henry Wilson</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>27042</td>
      <td>30.0</td>
      <td>A23</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Datasets are fine for analytics and machine learning, but when using in production, it is more convenient to have a more straightforward interface, that just gets the required `dataframe` without all these details. This interface is `DataFrameSource`, and `Dataset` can be converted to it:


```python
df_source = dataset.as_data_frame_source(columns=['Age','Sex','Survived'], count=3)
df_source.get_df()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38.0</td>
      <td>female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.0</td>
      <td>female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.0</td>
      <td>male</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Summary

In this section, the big, real-data datasets are covered:
* Creating a production-ready `FeaturizationJob` that creates dataset
* Accessing the created dataset with the `Dataset` class
* `FileSyncer` as the primary interface to syncronize the local data with the remote host
