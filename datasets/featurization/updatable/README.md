# 2.3.2 Updatable Featurization Jobs and Datasets (tg.common.datasets.featurization.updatable)

## Overview

If the data in the dataset are changing, we may want to update dataset from time to time. However, we don't want to reprocess all the records, as it takes too much time. This problem is addressed by the `UpdatableDataset` class and jobs that create and process it.

The dataset consists of _revisions_. Each revision is essentially a dataframe, that may be partitioned into smaller dataframes for optimization reasons (as done in `FeaturizationJob`). Rows of each revision describes the entities (like passengers, customers, or articles), and the dataframe index must be a unique identifier for the entity. If several rows are available in different revisions for the same entity, the later takes precedence. 

When stored in S3 or other external storages, several datasets may be stored together the same way as we have seen in the `FeaturizationJob`: the data are the same, but processed by different featurizers. 

Revisions may be _major_ and _minor_. Major revision corresponds to reevauation of all entities. It **does not** physically remove data from the dataset, but effectively voids them - data older that latest major revision are not read by default. The minor revisions are supposed to be small updates that rolls on top of the major revision. Due to this design:

* it is possible to restore the dataset's state to any given time in the past. 
* it is also possible not to carry around the very old data all the time, as they can be voided by a major revision.

The structure of the files on the S3 or other storages is:

```
root
↳ revision_1
  ↳ featurizer_X
    ↳ partition_1_X_i.parquet
    ↳ partition_1_X_ii.parquet
  ↳ featurizer_Y
    ↳ partition_1_Y_i.parquet
    ↳ partition_1_Y_ii.parquet
↳ revision_2
  ↳ featurizer_X
    ↳ partition_2_X_i.parquet
    ↳ partition_2_X_ii.parquet
  ↳ featurizer_Y
    ↳ partition_2_Y_i.parquet
    ↳ partition_2_Y_ii.parquet
↳ description.parquet
```

`revision_*` folders and `partition_*` files are  usually GUIDs, `featurizer_*` have meaningful names. 

`description.parquet` is the file describing the revisions. For each revision we know:

* the timestamp it was produced at
* if the revision was major or minor
* and the version of the job that has produced the revision. If the version was updated, the major revision should be triggered.

Also, `description.parquet` plays a role of transactions keeper: all the jobs producing datasets upload this file at the very end. Since this file keeps record of others, incomplete upload will not have any effect (aside from consuming space on the remote storage).

When working with this dataset, we specify the time `T` and download revisions between `T` and the last major revision before `T`. We also specify one featurizer, and download data for that. The resulting local file structure is:

```
root
↳ revision_1
  ↳ partition_1_X_i.parquet
  ↳ partition_1_X_ii.parquet
↳ revision_2
  ↳ partition_2_X_i.parquet
  ↳ partition_2_X_ii.parquet
↳ description.parquet
```

## Creating and reading updatable datasets

To demonstrate the file structure above, we will create an artificial dataset witout any particular meaning. The simplest way to do it is using a method in `UpdatableDataset` class. As before, we will use `MemoryFileSyncer` to demonstrate the file structure on the remote host.


```python
from tg.common.datasets.featurization import UpdatableDataset
from tg.common import MemoryFileSyncer
from datetime import datetime
import pandas as pd

mem = MemoryFileSyncer()

def time(day):
    return datetime(2020,1,1+day)

def create_dataframe(day, featurizer):
    N=5
    return pd.DataFrame(dict(
        day = [day] * N,
        featurizer = [featurizer] * N,
        id = list(range(day, day+N))
    )).set_index("id")

for day in [0,2,4,6]:
    record = UpdatableDataset.DescriptionItem(
        name = f'revision_{day}',
        timestamp = time(day),
        is_major = day%4==0,
        version = '')
    data = {featurizer: create_dataframe(day, featurizer) 
            for featurizer in ['featurizer_A','featurizer_B']}
    UpdatableDataset.write_to_updatable_dataset(
        syncer = mem,
        record = record,
        data = data
    )
```

Here is the list of files and folders, created in the `MemoryFileSyncer`:


```python
list(mem.cache)
```




    ['revision_0/featurizer_B/data.parquet',
     'revision_0/featurizer_A/data.parquet',
     'description.parquet',
     'revision_2/featurizer_B/data.parquet',
     'revision_2/featurizer_A/data.parquet',
     'revision_4/featurizer_B/data.parquet',
     'revision_4/featurizer_A/data.parquet',
     'revision_6/featurizer_B/data.parquet',
     'revision_6/featurizer_A/data.parquet']



The description of the dataset:


```python
pd.read_parquet(mem.get_file_stream('description.parquet'))
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>timestamp</th>
      <th>is_major</th>
      <th>version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>revision_0</td>
      <td>2020-01-01</td>
      <td>True</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>revision_2</td>
      <td>2020-01-03</td>
      <td>False</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>revision_4</td>
      <td>2020-01-05</td>
      <td>True</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>revision_6</td>
      <td>2020-01-07</td>
      <td>False</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



Here is how one revision looks like:


```python
pd.read_parquet(mem.get_file_stream('revision_0/featurizer_A/data.parquet'))
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>featurizer</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
  </tbody>
</table>
</div>



Here is the table, describing our dataset for `featurizer_A`: in columns, there are revisions, in rows, entities, and in cells, the `day` field of the dataframe. 


```python
dfs = []
for day in [0,2,4,6]:
    df = pd.read_parquet(mem.get_file_stream(f"revision_{day}/featurizer_A/data.parquet"))
    dfs.append(df[['day']].rename(columns={'day':f'day_{day}'}))
summary_df = pd.concat(dfs,axis=1)
summary_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day_0</th>
      <th>day_2</th>
      <th>day_4</th>
      <th>day_6</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



Use this table for reference: we will now read this dataset at specific time periods, so you will be able to see which records are available, and at which time.

Now, let's read the dataset with the `UpdatableDataset` class. It will "download" the data to the local drive (in this particular example, it downloads from memory).


```python
dataset = UpdatableDataset(
    location = './temp/updatable_dataset',
    featurizer_name = 'featurizer_A',
    syncer = mem
)

dataset.read(cache_mode='remake')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>featurizer</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>featurizer_A</td>
    </tr>
  </tbody>
</table>
</div>



**Note** the `download` argument. In the normal setup, we download dataset to the appropriate timeframe once with `download` method, and then just read from it. In this paragraph however we want to demonstrate downloading from different timeframes, so we will have to download with each read.

This is the "current" state of the dataset: 
* records 9-10 were added in the last minor revision (6)
* records 6-8 were updated in the last minor revision (6)
* records 4-5 were inherited from the revision (4)
* records 1-3 are lost, because the revision (4) is major

This is the list of local files:


```python
from yo_fluq_ds import *

Query.folder('./temp/updatable_dataset','**/*').foreach(print)
```

    temp/updatable_dataset/revision_4
    temp/updatable_dataset/description.parquet
    temp/updatable_dataset/revision_6
    temp/updatable_dataset/revision_4/featurizer_A
    temp/updatable_dataset/revision_4/featurizer_A/data.parquet
    temp/updatable_dataset/revision_6/featurizer_A
    temp/updatable_dataset/revision_6/featurizer_A/data.parquet


As we can see, indeed `UpdatableDataset` has only downloaded what is needed for the provided timeframe.

`UpdatableDataset` can restore it's state to any given point in the past:


```python
dataset.read(to_timestamp=time(1), cache_mode='remake')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>featurizer</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
  </tbody>
</table>
</div>



As we see, "lost" records 1-3 are available.


```python
dataset.read(to_timestamp=time(3), cache_mode='remake')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>featurizer</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>featurizer_A</td>
    </tr>
  </tbody>
</table>
</div>



Also, `UpdatableDataset` can provide you with changes, made between a given timestamps:


```python
dataset.read(from_timestamp=time(1), to_timestamp=time(3), cache_mode='remake')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>featurizer</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>featurizer_A</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.read(from_timestamp=time(1), to_timestamp=time(5), cache_mode='remake')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>featurizer</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>featurizer_A</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4</td>
      <td>featurizer_A</td>
    </tr>
  </tbody>
</table>
</div>



Between the days 1 and 5, two revisions took place: (2) and (4). But, since the revision (4) is major, it voids the output of the revision (2).

We encourage you to play with this code, downloading and reading data for different timeframes, and comparing with `summary_df` table.

`UpdatableDataset` supports `count`, `selector` and `columns` arguments for reading, as well as `as_data_frame_source` method that converts this dataset to the `DataFrameSource` interface. The guidelines of dataset exploration are the same as for `Dataset`.

## Updatable Featurization Job

`UpdableFeaturizationJob` is an analogue to `FeaturizationJob` that supports updates. The arguments of this class are essentially the same as for `FeaturizationJob`, because under the hood `UpdatableFeaturizationJob` only sorts out the timeframes, and spawns `FeaturizationJob` instances for actual featurization.

The only difference lies with `DataSource` that the job consumes. `UpdatableFeaturizationJob` requires you to provide a full data source that is used in major updates. If you want to have minor updates, you must also provide a factory that created a `DataSource` for a given timeframe.

Let's create an `UpdatableFeaturizationJob` for the Titanic dataset. Since this dataset doesn't actually have the time dependency, we will just split it into 3 groups, by `Embarked` field:
* for the initial major revision, `Embarked` will be set to None for all passengers (as if made before the trip started)
* following revisions will add the `Embarked` fields

The `Embarked` field is distributed as follows:


```python
df = pd.read_csv('titanic.csv')
df.groupby(df.Embarked.fillna("NONE")).size()
```




    Embarked
    C       168
    NONE      2
    Q        77
    S       644
    dtype: int64



The data sources are implemented as follows:


```python
from tg.common.datasets.access import DataSource

class EmbarkedDataSource(DataSource):
    def __init__(self, embarked):
        self.embarked = embarked
        
    def get_data(self):
        df = pd.read_csv('titanic.csv')
        if self.embarked is None:
            df.Embarked = 'NONE'
        else:
            df = df.loc[df.Embarked == self.embarked]
        return Query.df(df)
        
def source_factory(from_timestamp, to_timestamp):
    if to_timestamp.day == 3:
        return EmbarkedDataSource('C')
    elif to_timestamp.day == 5:
        return EmbarkedDataSource('Q')
    else:
        return EmbarkedDataSource('S')
```

Now we will implement and run `UpdatableFeaturizationJob`. Note that we must take some action regarding the dataframe indices: before we were simply ignoring it, but now we have to make sure that index is a unique identifier for an entity, i.e., passenger.


```python
from tg.common.datasets.featurization import UpdatableFeaturizationJob, DataframeFeaturizer

dataset_buffer = MemoryFileSyncer()

class PassengerFeaturizer(DataframeFeaturizer):
    def __init__(self):
        super(PassengerFeaturizer, self).__init__(row_selector=lambda z: z)
    
    def _postprocess(self, df):
        return df.set_index('PassengerId')
    
job = UpdatableFeaturizationJob(
    name = 'test_featurization_job',
    version = 'v1',
    full_data_source=EmbarkedDataSource(None),
    update_data_source_factory=source_factory,
    featurizers = dict(passengers = PassengerFeaturizer()),
    syncer = dataset_buffer,
    limit = None,
    reporting_frequency=None
)

for i in [0,2,4,6]:
    job.run(current_time = time(i),custom_revision_id=str(i))
```

    2022-06-27 19:28:46.523703+00:00 INFO: Starting lesvik job test_featurization_job, version v1
    2022-06-27 19:28:46.524324+00:00 INFO: Additional settings limit NONE, reporting NONE
    2022-06-27 19:28:46.524953+00:00 INFO: 0 previous revisions are found
    2022-06-27 19:28:46.525745+00:00 INFO: Running with id 0 at 2020-01-01 00:00:00, revision is MAJOR
    2022-06-27 19:28:46.527909+00:00 INFO: Featurization Job test_featurization_job at version v1 has started
    2022-06-27 19:28:46.533304+00:00 INFO: Fetching data
    2022-06-27 19:28:46.601873+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:28:46.616490+00:00 INFO: Uploading data
    2022-06-27 19:28:46.617495+00:00 INFO: Featurization job completed
    2022-06-27 19:28:46.617919+00:00 INFO: 891 were processed
    2022-06-27 19:28:46.618350+00:00 INFO: Uploading new description
    2022-06-27 19:28:46.621492+00:00 INFO: Job finished
    2022-06-27 19:28:46.621916+00:00 INFO: Starting lesvik job test_featurization_job, version v1
    2022-06-27 19:28:46.622640+00:00 INFO: Additional settings limit NONE, reporting NONE
    2022-06-27 19:28:46.626653+00:00 INFO: 1 previous revisions are found
    2022-06-27 19:28:46.627149+00:00 INFO: Running with id 2 at 2020-01-03 00:00:00, revision is MINOR
    2022-06-27 19:28:46.627524+00:00 INFO: Featurization Job test_featurization_job at version v1 has started
    2022-06-27 19:28:46.632681+00:00 INFO: Fetching data
    2022-06-27 19:28:46.645002+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:28:46.651358+00:00 INFO: Uploading data
    2022-06-27 19:28:46.652436+00:00 INFO: Featurization job completed
    2022-06-27 19:28:46.652897+00:00 INFO: 168 were processed
    2022-06-27 19:28:46.653307+00:00 INFO: Uploading new description
    2022-06-27 19:28:46.656803+00:00 INFO: Job finished
    2022-06-27 19:28:46.659087+00:00 INFO: Starting lesvik job test_featurization_job, version v1
    2022-06-27 19:28:46.659531+00:00 INFO: Additional settings limit NONE, reporting NONE
    2022-06-27 19:28:46.666662+00:00 INFO: 2 previous revisions are found
    2022-06-27 19:28:46.667215+00:00 INFO: Running with id 4 at 2020-01-05 00:00:00, revision is MINOR
    2022-06-27 19:28:46.667586+00:00 INFO: Featurization Job test_featurization_job at version v1 has started
    2022-06-27 19:28:46.671935+00:00 INFO: Fetching data
    2022-06-27 19:28:46.678229+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:28:46.683663+00:00 INFO: Uploading data
    2022-06-27 19:28:46.684565+00:00 INFO: Featurization job completed
    2022-06-27 19:28:46.684980+00:00 INFO: 77 were processed
    2022-06-27 19:28:46.685392+00:00 INFO: Uploading new description
    2022-06-27 19:28:46.688722+00:00 INFO: Job finished
    2022-06-27 19:28:46.689138+00:00 INFO: Starting lesvik job test_featurization_job, version v1
    2022-06-27 19:28:46.689487+00:00 INFO: Additional settings limit NONE, reporting NONE
    2022-06-27 19:28:46.698186+00:00 INFO: 3 previous revisions are found
    2022-06-27 19:28:46.699658+00:00 INFO: Running with id 6 at 2020-01-07 00:00:00, revision is MINOR
    2022-06-27 19:28:46.700191+00:00 INFO: Featurization Job test_featurization_job at version v1 has started
    2022-06-27 19:28:46.704410+00:00 INFO: Fetching data
    2022-06-27 19:28:46.755171+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:28:46.771760+00:00 INFO: Uploading data
    2022-06-27 19:28:46.774265+00:00 INFO: Featurization job completed
    2022-06-27 19:28:46.776489+00:00 INFO: 644 were processed
    2022-06-27 19:28:46.777205+00:00 INFO: Uploading new description
    2022-06-27 19:28:46.783469+00:00 INFO: Job finished


I have given a meaningful names to revisions with `custom_uid` argument. This is not necessary in general, as the order of revision is reflected in `description.parquet`.

Let's look at the revisions created. I will also add the information about records that were processed.


```python
from collections import OrderedDict



def get_embarkation_by_revision(buffer):
    desc_df = pd.read_parquet(buffer.get_file_stream('description.parquet'))
    rows = []
    for key in buffer.cache:
        if key == 'description.parquet':
            continue
        df = buffer.get_parquet(key)
        df = df.groupby('Embarked').size().to_frame().transpose().iloc[0]
        row = OrderedDict()
        row['partition'] = key.split('/')[0]
        row['file'] = key.split('/')[2]
        for s in Query.series(df):
            row[s.key] = s.value
        
        rows.append(row)
        
    edf = pd.DataFrame(rows)
    desc_df = desc_df.merge(edf.set_index('partition'), left_on='name',right_index=True)
    return desc_df

desc_df = get_embarkation_by_revision(dataset_buffer)
desc_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>timestamp</th>
      <th>is_major</th>
      <th>version</th>
      <th>file</th>
      <th>NONE</th>
      <th>C</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2020-01-01</td>
      <td>True</td>
      <td>v1</td>
      <td>8f0bab6c-df5b-4bc3-b6ab-2522cd787888.parquet</td>
      <td>891.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2020-01-03</td>
      <td>False</td>
      <td>v1</td>
      <td>be38c867-56e5-458b-9a85-e5518ef4f025.parquet</td>
      <td>NaN</td>
      <td>168.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2020-01-05</td>
      <td>False</td>
      <td>v1</td>
      <td>39ffd0ae-63f7-4644-964d-2e38998fe668.parquet</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>2020-01-07</td>
      <td>False</td>
      <td>v1</td>
      <td>28346ab8-5aa2-431b-9484-5cba9a30f454.parquet</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>644.0</td>
    </tr>
  </tbody>
</table>
</div>



As we see, all worked as expected.

The created dataset can be then accessed via `UpdatableDataset` class.


```python
dataset = UpdatableDataset(
    location = './temp/updatable_dataset_2',
    featurizer_name = 'passengers',
    syncer = dataset_buffer
)
df = dataset.read(cache_mode='remake', partition_name_column='partition_name')
df = df.merge(
    dataset.get_desription_as_df().set_index('name'),
    left_on='partition_name',
    right_index=True)
df.groupby(['Embarked','timestamp']).size().to_frame('records_count').reset_index()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>timestamp</th>
      <th>records_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>2020-01-03</td>
      <td>168</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NONE</td>
      <td>2020-01-01</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Q</td>
      <td>2020-01-05</td>
      <td>77</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S</td>
      <td>2020-01-07</td>
      <td>644</td>
    </tr>
  </tbody>
</table>
</div>



The handy argument `partition_name_column` adds to each rows the name of the revision that has produced this row. This name may be then used to merge with the description for further technical information.

## `UpdatableDatasetScoringJob`

There are probably lots of scenarios how the UpdatableDatasets may be processed. So far, we have suppored one scenario, `UpdatableDatasetScoringJob`. This job is mostly designed to compute scores for entities in datasets for different purposes. So the actions performed are:

* Get the last time when the job was run
* Download updates from dataset(s) from this last time
* Compute scores for the update
* Upload the updates for the dataset

Such job consists of UpdatableDatasetScoringMethods, each method knows:
* The dataset it pulls data from
* The function that must be applied to the resulting dataframe. The function must preserve the dataframe index.

Let's define such scoring method and job.


```python
from tg.common.datasets.featurization import UpdatableDatasetScoringJob, UpdatableDatasetScoringMethod
from datetime import timedelta

def compute_scores(df):
    return df[['Survived','Embarked']]

scores_buffer = MemoryFileSyncer()

job = UpdatableDatasetScoringJob(
    name = 'scoring',
    version = '',
    dst_syncer = scores_buffer,
    methods = [
        UpdatableDatasetScoringMethod(
            'passenger_scores',
            dataset_buffer,
            'passengers',
            compute_scores        
        )
    ]
)

for i in [0,2,4,6]:
    job.run(current_time = time(i)+timedelta(hours=1), custom_revision_id=str(i))
```

The addition of one hour is required to reflect the fact that the scoring job would need to run _after_ the initial job. The 


```python
get_embarkation_by_revision(scores_buffer)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>timestamp</th>
      <th>is_major</th>
      <th>version</th>
      <th>file</th>
      <th>NONE</th>
      <th>C</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2020-01-01 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>07091c46-a2e3-452d-b1c8-2ed5bdca4c11.parquet</td>
      <td>891.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2020-01-03 01:00:00</td>
      <td>False</td>
      <td></td>
      <td>2572de70-39c4-432c-86b9-99c480749a57.parquet</td>
      <td>NaN</td>
      <td>168.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2020-01-05 01:00:00</td>
      <td>False</td>
      <td></td>
      <td>a2e88c8a-4f4d-4936-bf49-f1d23de346b5.parquet</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>2020-01-07 01:00:00</td>
      <td>False</td>
      <td></td>
      <td>f8bc5246-c2c6-4305-bf3a-5f911c2c8087.parquet</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>644.0</td>
    </tr>
  </tbody>
</table>
</div>



Again, as we see, all worked as expected: each scoring job instance have only processed the data from the revision that was unprocessed at the time.

Let's change this behavior for demonstration purposes. With the `custom_start_time` argument, I will force the job to obtain all the changes made in the initial dataset since the time 0, not since that last run of the job.


```python
scores_buffer = MemoryFileSyncer()

job = UpdatableDatasetScoringJob(
    name = 'scoring',
    version = '',
    dst_syncer = scores_buffer,
    methods = [
        UpdatableDatasetScoringMethod(
            'passenger_scores',
            dataset_buffer,
            'passengers',
            compute_scores        
        )
    ]
)

for i in [0,2,4,6]:
    job.run(
        current_time = time(i)+timedelta(hours=1), 
        custom_revision_id=str(i),
        custom_start_time=time(0)
    )
```


```python
get_embarkation_by_revision(scores_buffer)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>timestamp</th>
      <th>is_major</th>
      <th>version</th>
      <th>file</th>
      <th>NONE</th>
      <th>C</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2020-01-01 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>3505ad46-d75c-41c1-8bb0-92f0cbb3bfcb.parquet</td>
      <td>891.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2020-01-03 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>c1314e6e-e5e3-4af0-9acc-d1fca409c605.parquet</td>
      <td>NaN</td>
      <td>168.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2020-01-03 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>2c793de3-9046-4f08-9cf4-2881cb46f501.parquet</td>
      <td>723.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2020-01-05 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>e2029fec-a0bf-400d-a296-827343773dd3.parquet</td>
      <td>646.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2020-01-05 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>02396733-33cc-40b0-9d5e-2751677c334d.parquet</td>
      <td>NaN</td>
      <td>168.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2020-01-05 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>3cf02629-f21a-48d9-ad38-6db68a59b343.parquet</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>2020-01-07 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>f882fdec-0c0d-475b-8df5-c4760b48a410.parquet</td>
      <td>NaN</td>
      <td>168.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>2020-01-07 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>62b4c9cb-4eda-4ee4-be4a-5d3798b2f1ef.parquet</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>2020-01-07 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>811c8db6-3769-45f3-ba65-69979d351e74.parquet</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>644.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>2020-01-07 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>7cd03fe5-bef1-4df6-a23b-6281b74fcdb1.parquet</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Why are there so many files? Why are the different embarkations always in different files?

Because when we read changes from the revisions 0, 2, and 4, there are 3 physical files to read (1 at each revisions). We don't know for sure how these files are organized: maybe they override records of the previous ones, but _maybe_ they just add records, and _maybe_ if we try to read them all at once, we will overload the memory. This is why instead of `read` method,  `read_iter` is used. For the (4) revision, it yields: 

* first the dataframe from the revision (4), 
* then the dataframe from the revision (2), minus records in revision (4)
* then the dataframe from the revision (0) minus records in revision (2), (0)

If the data would be partitioned, it would further increase amount of dataframes.

Each of these dataframes has an important property: it is not bigger that something, created by `FeaturizationJob`. Therefore, it will not overload the memory. So regardless of how long this system is run untouched, the data accumulation will not result in memory overflow. This approach should be applied to all other jobs that process `UpdatableDatasets`



