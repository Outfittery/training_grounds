# 2.1. Data Sources (tg.common.datasets.access)

## Overview

Training Ground offers the Data Objects Flow (DOF) model as the primary interface to access the data: 

* *Data object* are non-relational JSONs, typically huge and containing nested lists and dictionaries. 
* *Flow* means that objects are not placed in memory all at once, but are accessible as a python iterator.

`DataSource` is the primary interface that hides the database implementation and exposes data in DOF format: `get_data` method returns an interator, wrapped as `Queryable` class from `yo_fluq_ds` (https://pypi.org/project/yo-fluq-ds/). `DataSource` is a necessary abstraction that hides the details of how the data are actually stored: be it relational database, AWS S3 storage or simply a file, as long as the data can be represented as a flow of DOFs, you will be able to use it in your project. If the data storage changes, you may adapt to this change by replacing the `DataSource` implementation and keeping the rest of the featurization process intact. Typically, you need to implement your own `DataSource` inheritants for the storages you have in your environment.

The goal of featurization is typically converting DOF into a tidy dataframe. In this demo, we will work with the well-known Titanic dataset, which is stored in the local folder as a `csv` file. Of course, it already contains all the data in the tidy format, but for the sake of the demonstration we will distort this format. In the following demos, the tidiness will be restored again with the TG-pipeline. 





## DataSource 


The first step is to write your own `DataSource` class, that will make Titanic dataset available as DOF.


```python
from yo_fluq_ds import Query, Queryable
from tg.common.datasets.access import DataSource
import pandas as pd

class CsvDataSource(DataSource):
    def __init__(self, filename):
        self.filename = filename

    def _get_data_iter(self):
        df = pd.read_csv(self.filename)
        for row in df.iterrows():
            d = row[1].to_dict()
            yield  {
                'id': d['PassengerId'],
                'ticket': {
                    'ticket.id': d['Ticket'],
                    'fare': d['Fare'],
                    'Pclass': d['Pclass']
                },
                'passenger': {
                    'Name': d['Name'],
                    'Sex': d['Sex'],
                    'Age': d['Age']
                },
                'trip': {
                    'Survived': d['Survived'],
                    'SibSp': d['SibSp'],
                    'Patch': d['Parch'],
                    'Cabin': d['Cabin'],
                    'Embarked' : d['Embarked']
                    
                }
            }
            
    def get_data(self) -> Queryable:
        return Query.en(self._get_data_iter())
    
source = CsvDataSource('titanic.csv')

for row in source.get_data():
    print(row)
    break
```

    {'id': 1, 'ticket': {'ticket.id': 'A/5 21171', 'fare': 7.25, 'Pclass': 3}, 'passenger': {'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male', 'Age': 22.0}, 'trip': {'Survived': 0, 'SibSp': 1, 'Patch': 0, 'Cabin': nan, 'Embarked': 'S'}}


Here `_get_data_iter` creates an iterator, that yields objects one after another. In `get_data`, we simply wrap this iterator in `Queryable` type from `yo_fluq`. It's still the iterator, so we can iterate over it, as `for` loop demonstrates.

`Queryable` class contains a variety of methods for easy-to-write data processing, which are the Python-port of LINQ technology in C#. The methods are explained in full details in https://pypi.org/project/yo-fluq-ds/ . The access to the DOF in `Queryable` format allows you to quickly perform exploratory data analysis. As an example, consider the following code:


```python
(source
 .get_data()
 .where(lambda z: z['passenger']['Sex']=='male')
 .order_by(lambda z: z['passenger']['Age'])
 .select(lambda z: z['ticket'])
 .take(3)
 .to_dataframe()
)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticket.id</th>
      <th>fare</th>
      <th>Pclass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2625</td>
      <td>8.5167</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>250649</td>
      <td>14.5000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>248738</td>
      <td>29.0000</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



The meaning is self-evident: filter by `Sex`, order by `Age` and select the `Ticket` information out of the records, then take 3 of them in the format of pandas `DataFrame`.

## Caches

Quite often we want to make the data available offline, so the data is available faster and do not create a load on the external server. The typical use cases are:

* Exploratory data analysis
* Functional tests in your service: these tests often use the real data, and it's impractical to wait each time for this data to be delivered.
* Debugging of you services: most of the data services are downloading some data at the beginning, and in order to speed-up the startup when debugging on the local machine, it's helpful to create a cache.

To make data source cacheable, create a wrapper:


```python
from tg.common.datasets.access import ZippedFileDataSource, CacheableDataSource

cacheable_source = CacheableDataSource(
    inner_data_source = source,
    file_data_source = ZippedFileDataSource(path='./temp/source/titanic')
)
```

`CacheableDataSource` is still a `DataSource` and can be called directly. In this case, the original source will be called.


```python
cacheable_source.get_data().first()
```




    {'id': 1,
     'ticket': {'ticket.id': 'A/5 21171', 'fare': 7.25, 'Pclass': 3},
     'passenger': {'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male', 'Age': 22.0},
     'trip': {'Survived': 0,
      'SibSp': 1,
      'Patch': 0,
      'Cabin': nan,
      'Embarked': 'S'}}



However, we can also access data this way:


```python
from tg.common.datasets.access import CacheMode

cacheable_source.safe_cache(CacheMode.Default).get_data().first()
```




    {'id': 1,
     'ticket': {'ticket.id': 'A/5 21171', 'fare': 7.25, 'Pclass': 3},
     'passenger': {'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male', 'Age': 22.0},
     'trip': {'Survived': 0,
      'SibSp': 1,
      'Patch': 0,
      'Cabin': nan,
      'Embarked': 'S'}}



You can also use a string constant for this:


```python
cacheable_source.safe_cache('default').get_data().first()
```




    {'id': 1,
     'ticket': {'ticket.id': 'A/5 21171', 'fare': 7.25, 'Pclass': 3},
     'passenger': {'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male', 'Age': 22.0},
     'trip': {'Survived': 0,
      'SibSp': 1,
      'Patch': 0,
      'Cabin': nan,
      'Embarked': 'S'}}



`safe_cache` accepts the following modes: 
* `CacheMode.Default/default` mode, in this case `safe_cache` will create the cache in the `path` folder, provided to `ZippedFileDataSource`, if it does not exists, and read from it. 
* `CacheMode.Use/use` mode. the error will be thrown if cache does not exist locally. 
* `CacheMode.No/no` mode, the underlying source will be called directly, the cache will neither created nor used.
* `CacheMode.Remake/remake` forces the cache to be created even if it already exists.

So, when developing, we can use caches to save time, but when deploying, disable caching them with simple change of the argument. 

The format for the created cache file is a zipped folder with files that contains pickled data separated into bins. Normally, you don't need to intervene to their size. Increasing the bins size increases both performance and memory consumption. Theoretically, you may use another format by implementing your own class instead of `ZippedFileDataSource`. However, it's only recommended: the current format is a result of a comparative research, and other, more obvious ways of caching (caching everything in one file, or caching each object in an invidual file) perform much slower.

## Additional use of `CacheMode`

In practice, `CacheMode` often becomes a single argument to the whole data aquisition component of the application: 
* `no` is used for the production run
* `default` for local debugging, this way all the nesessary data is cached and starting the application up is much faster
* `remake` if you want to update the local data
* `use` in integration tests, which you want to run quickly and exactly on the same data the local application is running

Since data aquisition may sometimes go without `DataSource` class, the following method is created:


```python
from uuid import uuid4

def create_data():
    return str(uuid4())

uid1 = CacheMode.apply_to_file(
    CacheMode.Default,
    './temp/cached_data',
    create_data
)

uid2 = CacheMode.apply_to_file(
    CacheMode.Default,
    './temp/cached_data',
    create_data
)

uid3 = CacheMode.apply_to_file(
    CacheMode.No,
    './temp/cached_data',
    create_data
)

uid1, uid2, uid3
```




    ('92cd4fb0-7d6f-4633-8bdb-b8976cb0cff8',
     '92cd4fb0-7d6f-4633-8bdb-b8976cb0cff8',
     '5ee75250-6fcf-4772-b52e-d5046cfba58e')



First time dataframe will be created by `create_data` method, but for the second time, it will be read from the file, so `uid1` and `uid2` are the same, `uid3` is different, because `CacheMode.No` was used as an argument.

In the particular case of data in `pandas` dataframe format, TG also offers `DataFrameSource` interface.


```python
from tg.common.datasets.access import DataFrameSource

class TestDataFrameSource(DataFrameSource):
    def get_df(self):
        return pd.DataFrame([dict(a=1,b=2)])
    
src = TestDataFrameSource()
src.get_df()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



The `get_cached_df` method wraps `CacheMode.apply_to_file` method and allows you to cache dataframe quicker.

## Summary

In this demo, we have covered the following topics:
* Data Objects Flow as the primary model of incoming data in TG fearurization process
* `DataSource` as the main interface providing DOF
* Caching `DataSource` with `CacheableDataSource` and `CacheMode`
* Caching other types of data

