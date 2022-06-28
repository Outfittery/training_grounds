# 1. Overview and installation instructions (tg.common)

## Overview

Training Grounds (TG) is a data science framework, developed at [Outfittery Gmbh](http://www.outfittery.de).
Currently, it is used in several projects for data preprocessing pipelines and machine learning pipelines design.
The main tools available are:

1. `tg.common._common`: framework-wide classes, that are also importable directly from `tg.common`
1. `tg.common.datasets`, datasets and featurization: easy-to-write pure functions that convert the unstructured data into tidy dataset.
1. `tg.common.ml`, machine learning: SOLID-based architecture to define machine-learning pipelines, including test/train splitting, training itself, evaluation and hyperparameter tuning.
1. `tg.common.delivery`: the way to package the code, send it to the remote server and execute there. Here, TG provides a flexible way of delivery (no integration with the version control system or any other external software is required), 
reliable versioning technique and hot-module replacement.
1. `tg.common.analysis`: tools for analytics, mostly around statistical significance

Our main principles were:

1. Use the existing solution to the maximum extent and write the lightest wrappers possible.
1. SOLID to provide testability, extendability and reusability of the solutions.
1. "Open framework": we do not claim that we offer all components you will ever need; 
rather, we offer you the architecture where you can build components for your use cases, and the components we used for ours.
1. Minimizing complexity gap: instead of domain specific languages or visual programming that are often easy in the training phase, but are hard to adopt for the real use cases, not covered by the manual, we offer a code-first approach that behaves 
the same way on all the stages of adoption.

This repository contains:

1. The source code of the framework
1. The comprehensive set of Demos that covers the platform functionality. Demos are Jupyter Notebooks, so you can not only run them, but also play around and explore the functionality yourself. These demos are also compiled in `md` format and are available as `README.md` in the corresponding folders.
1. Tests that cover substantial part of the code, especially in the crucial parts, but not 100% of it.

This repository **is not** a Python module. This is due to technical requirements of the delivery subsystem, which
is covered in the Demo in full details. Two installation options are available.


## Fast installation

You might install the publically available [training grounds](https://pypi.org/project/training-grounds/) published in pypi.

Run

```
pip install training-grounds
```

in your environment. Run

```
jupyter notebook
```

in terminal, open the Demos in `tg/common/demos` folder.

On your local machine, you will be able to fully use TG without any limitations. As for delivery,
you will only be able to deliver the objects that are composed entirely from build-in TG classes.
This limitation is not something you should necessarily be concerned at your first steps.


## Full-fledged installation

Alternatively to the publically available version of `tg` you might install the the latest version of `tg` as submodule into your project.

Create a git repository for your project. We will refer to the folder of the repository as `/`. If you don't want to use `git`, just create `/` folder.

Add Training Grounds as a submodule to `tg/common` folder.

```
git submodule add git@github.com:outfittery/training_grounds.git tg/common
```

If you don't want to use `git`, just download TG repository to `tg/common` folder. You can choose another name for the folder instead of `tg/common`, but the tests and demos won't work in this case

Create an empty `tg/__init__.py`. Otherwise `tg` will not be recognized as module, so you won't be able to import from `tg.common`

Create a file `setup.py`. You may use the following template:

```.python
from setuptools import setup, find_packages
setup(name='Env for TG demo',
      version='0.0.0',
      description='Demo',
      packages=find_packages(),
      install_requires=[],
      include_package_data = True,
      zip_safe=False
)
```

Add to the `setup.py` the requirements for TG that are listed in `tg/common/requirements.txt`. You don't have to install all of them, e.g. if you don't use machine learning in the project, you don't have to install torch or sklearn.

Create and activate the virtual environment 

* E.g., via Anaconda:

    ```
    conda create --name tg python=3.8
    conda activate tg
    ```

* or virtualenv

    ```
    python3.8 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    ```

Go to `/` directory in Terminal and execute

```
pip install -e .
```

Execute

```
pip install -r tg/common/requirements.txt
```

in order to install all packages used in `tg`. To start the demos, type

```
jupyter notebook
```

in terminal, open the Demos in `tg/common/demos` folder.
 
### Tests

Tests are located in `tg/common/test_common`. Some of them use external dependencies:

* Docker must be installed and accessible without `sudo`
 
### Folders

There are two folders created by TG in your installation root directory:

* `temp` It contains files that can be deleted at any time.
* `data-cache`. It contains downloaded data. Deleting it will cause re-downloading.

These folders should be added to `.gitignore`. To access them from within code, use `tg.common.Loc` object.

### Custom code

If you wish to customize the TG classes for your project and keep them deliverable, you should create `tg/your_project` folder and place your code there.



# 1.1. Common features (tg.common._common)

`tg.common` defines a few auxiliary classes, that are used through the whole framework.

To prevent circular dependencies, the code is actually defined in `tg.common._common`. Not everything is exposed to the `tg.common`, as we want to limit the amount of universally exposed classes to the minimum.

## Logger

`Logger` is a slim wrapper over standard `logging` module, designed to augment the logging messages with "keys" that additionally describe the message. The main reason for this is integration with i.e. Kibana. 


These fields belong to one of the categories:
  * Automatic: code file, line, exception type, value and stacktrace
  * Base: the name of the service, version, etc.
  * Session: the user-defined keys.

Logger will work without any additional initialization, just by importing:


```python
from tg.common import Logger

Logger.info('Message with default logger')
```

    2022-06-27 19:37:48.287262+00:00 INFO: Message with default logger



```python
Logger.initialize_kibana()
Logger.info('Message with Kibana logger')
```

    {"@timestamp": "2022-06-27 19:37:48.292745+00:00", "message": "Message with Kibana logger", "levelname": "INFO", "logger": "tg", "path": "/tmp/ipykernel_1236/2907404717.py", "path_line": 2}


As said before, you may define a custom session keys:


```python
Logger.push_keys(test_key='test')
Logger.info('Message with a key')
Logger.clear_keys()
Logger.info('Message without a key')
```

    {"@timestamp": "2022-06-27 19:37:48.300363+00:00", "message": "Message with a key", "levelname": "INFO", "logger": "tg", "path": "/tmp/ipykernel_1236/71300885.py", "path_line": 2, "test_key": "test"}
    {"@timestamp": "2022-06-27 19:37:48.301611+00:00", "message": "Message without a key", "levelname": "INFO", "logger": "tg", "path": "/tmp/ipykernel_1236/71300885.py", "path_line": 4}


If exception information is available, it will be put in the keys:


```python
try:
    raise ValueError('Error')
except: 
    Logger.error('Error')
```

    {"@timestamp": "2022-06-27 19:37:48.306262+00:00", "message": "Error", "levelname": "ERROR", "logger": "tg", "path": "/tmp/ipykernel_1236/1975102656.py", "path_line": 4, "exception_type": "<class 'ValueError'>", "exception_value": "Error", "exception_details": "Traceback (most recent call last):\n  File \"/tmp/ipykernel_1236/1975102656.py\", line 2, in <module>\n    raise ValueError('Error')\nValueError: Error\n"}


To change the default way of logging, inherit `tg.common._common.logger.LoggerRoot` in `tg.your_project`, and then import from there. It will also affect all the logging within `TG` framework.

## Loc

In various situations, `TG` stores intermediate or cache files on your machine, typically in:
* `/temp` folder, these files do not require large efforts to create and thus you can delete it as often as you like.
* `/data-cache`: datasets and data downloads are stored there, and so restoring these files may take awhile.

To access these files:


```python
from tg.common import Loc
from yo_fluq_ds import FileIO

FileIO.write_text('test', Loc.temp_path/'test.txt')
FileIO.read_text(Loc.temp_path/'test.txt')
```




    'test'





# 2. Datasets (tg.common.datasets)

## Overview

`tg.common.datasets` contains the following modules:

* `tg.common.datasets.access`
  - Abstraction for connection to SQL databases and other datasources
  - Caching the data locally
* `tg.common.datasets.selectors`
  - Handly classes to convert unstructured data into tidy datasets
* `tg.common.datasets.featurization`
  - Production-ready code to get the data, convert into tidy datasets and store at the external storage



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




    ('841c3d4a-c8e2-4791-b43b-113f6dc74725',
     '841c3d4a-c8e2-4791-b43b-113f6dc74725',
     '97eafd64-4544-4982-a682-4ee221c8d6f7')



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




# 2.2. Selectors (tg.common.datasets.selectors)

## Overview

Selectors are objects that:
* define a pure function that transforms a data object into a row in the dataset
* track errors and warnings that happen during this conversion
* fully maintain the inner structure of selectors, making it possible to e.g. visualize the selector

**Note**: selectors are slow! They are not really aligned for the processing of hundreds of gygabytes of data. If this use case arises:
* they potentially can be parallelized in a PySpark 
* they potentially can be partially translated into e.g. PrestoSQL queries, since they maintain the inner structure


## Basic Selectors

We will use the distorted Titanic dataset from the previous demo, and apply various selectors to one of the Data Objects.


```python
from tg.common.datasets.access import ZippedFileDataSource, CacheableDataSource, CacheMode

source = CacheableDataSource(
    inner_data_source = None,
    file_data_source = ZippedFileDataSource(path='./titanic.zip'),
    default_mode=CacheMode.Use
)
obj = source.get_data().skip(11).first()
obj
```




    {'id': 12,
     'ticket': {'ticket.id': '113783', 'fare': 26.55, 'Pclass': 1},
     'passenger': {'Name': 'Bonnell, Miss. Elizabeth',
      'Sex': 'female',
      'Age': 58.0},
     'trip': {'Survived': 1,
      'SibSp': 0,
      'Patch': 0,
      'Cabin': 'C103',
      'Embarked': 'S'}}



Let's start with simply selecting one field:


```python
from tg.common.datasets.selectors import Selector

selector = (Selector()
            .select('id')
            )

selector(obj)
```




    {'id': 12}



`Selector` class is a high-level abstraction, that allows you defining the featurization function with a `Fluent API`-interface. `Selector` is building a complex object of interconnected smaller processors, and we will look at these processors a little later. We may consider `Selector` on a pure syntax level: how exactly this or that use case can be covered with it. 

We can rename the field as follow:


```python
selector = (Selector()
            .select(passenger_id = 'id')
            )

selector(obj)
```




    {'passenger_id': 12}



We can select nested fields several syntax options:


```python
from tg.common.datasets.selectors import Selector, FieldGetter, Pipeline

selector = (Selector()
            .select(
                'passenger.Name',
                ['passenger','Age'],
                ticket_id = ['ticket',FieldGetter('ticket.id')],
                sex = Pipeline(FieldGetter('passenger'), FieldGetter('Sex'))
            ))
selector(obj)
```




    {'ticket_id': '113783',
     'sex': 'female',
     'Name': 'Bonnell, Miss. Elizabeth',
     'Age': 58.0}



* the first one (for `Name`) represents the highest level of abstraction, it is very easy to define lots of fields for selection in this way.
* the second one (for `Age`) shows that arrays can be used instead of dotted names. The elements of array will be applied sequencially to the input. In this particular case the array consists of two strings, and strings are used as the keys to extract values from dictionaries. Therefore, first the `passenger` will be extracted from the top-level dictionary, and then -- `Age` from `passenger`.
* the third (for `ticket.id`) is the only way how we can access the fields with the symbol `.` in name. `FieldGetter` is one of aforementioned small processors: it processes the given object by extracting the element out of the dictionary, or a field from the Python object. 
* the fourth way (for `Sex`) fully represents how selection works under the hood: it is a sequencial application (`Pipeline`) of two `FieldGetters`. So the arrays for `Age` and `ticket.id` will be converted to `Pipeline` under the hood.

The best practice is to use the first method wherever possible, and the third one in other cases.

If you select several fields from the same nested object, please use `with_prefix` method for optimization:


```python
from tg.common.datasets.selectors import Selector, FieldGetter, Pipeline

selector = (Selector()
            .with_prefix('passenger')
            .select('Name','Age','Sex')
            .select(ticket_id = ['ticket',FieldGetter('ticket.id')])
            )
selector(obj)
```




    {'Name': 'Bonnell, Miss. Elizabeth',
     'Age': 58.0,
     'Sex': 'female',
     'ticket_id': '113783'}



`with_prefix` method only affects the `select` that immediately follows it. 

Often, we need to post-process the values. For instance, name by itself is not likely to be feature (and would be GDPR-protected for the actual customers, thus making the entire output dataset GDPR-affected, which is better to avoid). However, we can extract the title from name as it can indeed be a predictor.


```python
import re

def get_title(name):
    title = re.search(' ([A-Za-z]+)\.', name).group().strip()[:-1]
    if title in ['Lady', 'Countess','Capt', 'Col',
                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
        return 'Rare'
    elif title == 'Mlle':
        return 'Miss'
    elif title == 'Ms':
        return 'Miss'
    elif title == 'Mme':
        return 'Mrs'
    else:
        return title

string_size = (Selector()
               .select(['passenger','Name', get_title])
              )

string_size(obj)
```




    {'Name': 'Miss'}



## Adress

Selectors are used to retrieve many fields from the input. Sometimes, we only want to retrieve one field, but with this handy interface. To do that, TG offers `Address` class


```python
from tg.common.datasets.selectors import Address

Address('passenger','Name', get_title)(obj)
```




    'Miss'



Or, the equivalent:


```python
Address.on(obj)('passenger','Name',get_title)
```




    'Miss'



`Address` is extremely useful for exploratory data analysis: by adding and removing arguments, you may move on the object forth and back:


```python
Address.on(obj)(list)
Address.on(obj)('passenger',list)
Address.on(obj)('passenger','Age')
Address.on(obj)('passenger','Age', type)
pass
```

## Ensembles and Pipelines

`Ensemble` and `Pipeline` classes are the machinery behind the `Selector` and `Address`, and ofter they are used directly, alongside with `Selector`. For instance, what if we want to apply multiple featurizers to the same field? We can use `Ensemble` for that:


```python
from tg.common.datasets.selectors import Ensemble

def get_length(s):
    return len(s)

string_size = (Selector()
               .select([
                   'passenger',
                   'Name', 
                   Ensemble(title=get_title, length=get_length)
               ]))


string_size(obj)
```




    {'Name': {'title': 'Miss', 'length': 24}}



`Ensemble` can also be used to combine several selectors together. When your Data Objects are huge and complicated, it makes more sense to write several smaller selectors, instead of writing one that selects all the fields you have. It's easier to read and reuse this way.


```python
ticket_selector = (Selector()
                   .with_prefix('ticket')
                   .select(
                       'fare',
                       'PClass',
                       id = [FieldGetter('ticket.id')]
                   )
)
passenger_selector = (Selector()
                      .with_prefix('passenger')
                      .select(
                          'Sex',
                          'Age',
                          name=['Name', Ensemble(
                              length=get_length,
                              title=get_title
                          )]
                      ))

combined_selector = Ensemble(
    ticket = ticket_selector,
    passenger = passenger_selector,
)
combined_selector(obj)
```

    2022-06-27 19:28:38.748037+00:00 WARNING: Missing field in FieldGetter





    {'ticket': {'id': '113783', 'fare': 26.55, 'PClass': None},
     'passenger': {'name': {'length': 24, 'title': 'Miss'},
      'Sex': 'female',
      'Age': 58.0}}



Pipelines, too, can be used for combination purposes. The typical use case is postprocessing: at the first step, we select fields from the initial object, and then, we want to compute some functions from these fields (e.g., we may want to compute BMI for the person from their weight and height). 

In Titanic example, let's compute a sum of `SibSp` and `Patch` as a new feature, `Relatives`. We will place it into the new `trip_selector` (which is selector, describing the trip in general, rather than the person or the ticket).

For that, we will use `Pipeline`. The arguments of the `Pipeline` are functions, that will be sequencially applied to the input.


```python
def add_relatives_count(d):
    d['Relatives'] = d['SibSp'] + d['Patch']
    return d

trip_selector = Pipeline(
    Selector()
     .select('id')
     .with_prefix('trip')
     .select('Survived','Cabin','Embarked','SibSp','Patch'),
    add_relatives_count
)

trip_selector(obj)
```




    {'id': 12,
     'Survived': 1,
     'Cabin': 'C103',
     'Embarked': 'S',
     'SibSp': 0,
     'Patch': 0,
     'Relatives': 0}



Now we need to do some finishing stitches: 
* for a problemless conversion to dataframe, we need a flat `dict`, not nested. TG has the method for that, `flatten_dict`
* We will also insert the current time as a processing time.


```python
from tg.common.datasets.selectors import flatten_dict
import datetime

def add_meta(obj):
    obj['processed'] = datetime.datetime.now()
    return obj

titanic_selector = Pipeline(
    Ensemble(
        passenger = passenger_selector,
        ticket = ticket_selector,
        trip = trip_selector
    ),
    add_meta,
    flatten_dict
)
titanic_selector(obj)
```

    2022-06-27 19:28:38.768354+00:00 WARNING: Missing field in FieldGetter





    {'passenger_name_length': 24,
     'passenger_name_title': 'Miss',
     'passenger_Sex': 'female',
     'passenger_Age': 58.0,
     'ticket_id': '113783',
     'ticket_fare': 26.55,
     'ticket_PClass': None,
     'trip_id': 12,
     'trip_Survived': 1,
     'trip_Cabin': 'C103',
     'trip_Embarked': 'S',
     'trip_SibSp': 0,
     'trip_Patch': 0,
     'trip_Relatives': 0,
     'processed': datetime.datetime(2022, 6, 27, 21, 28, 38, 769388)}



## Representation

The selectors always keep the internal structure and thus can be analyzed and represented in the different format. The following code demonstrates how this structure can be retrieved. 


```python
from tg.common.datasets.selectors import CombinedSelector
import json

def process_selector(selector):
    if isinstance(selector, CombinedSelector):
        children = selector.get_structure()
        if children is None:
            return selector.__repr__()
        result = {} # {'@type': str(type(selector))}
        for key, value in children.items():
            result[key] = process_selector(value)
        return result
    return selector.__repr__()


representation = process_selector(titanic_selector)

print(json.dumps(representation, indent=1)[:300]+"...")
            
```

    {
     "0": {
      "passenger": {
       "0": {
        "0": {
         "0": "[?passenger]"
        },
        "1": {
         "name": {
          "0": "[?Name]",
          "1": {
           "length": "<function get_length at 0x7fd80a3c2040>",
           "title": "<function get_title at 0x7fd80a428ca0>"
          }
         },
         "Sex": {
          "0": "...


To date, we didn't really find out the format that is both readable and well-representative, so we encourage you to explore and extend the code for representation creation to add the field you need for the effective debugging.

## Error handling

Sometimes selectors throw an error while processing the request. They provide a powerful tracing mechanism to find the cause of error in their complicated structure, as well as in the original piece of data.

Let us create an erroneous object for processing. The `Name` field which is normally a string, will be replaced with integer value.


```python
err_obj = source.get_data().first()
err_obj['passenger']['Name'] = 0
err_obj
```




    {'id': 1,
     'ticket': {'ticket.id': 'A/5 21171', 'fare': 7.25, 'Pclass': 3},
     'passenger': {'Name': 0, 'Sex': 'male', 'Age': 22.0},
     'trip': {'Survived': 0,
      'SibSp': 1,
      'Patch': 0,
      'Cabin': nan,
      'Embarked': 'S'}}




```python
from tg.common.datasets.selectors import SelectorException
exception = None
try:
    titanic_selector(err_obj)
except SelectorException as ex:
    exception = ex
    
print(exception.context.original_object)
print(exception.context.get_data_path())
print(exception.context.get_code_path())

```

    {'id': 1, 'ticket': {'ticket.id': 'A/5 21171', 'fare': 7.25, 'Pclass': 3}, 'passenger': {'Name': 0, 'Sex': 'male', 'Age': 22.0}, 'trip': {'Survived': 0, 'SibSp': 1, 'Patch': 0, 'Cabin': nan, 'Embarked': 'S'}}
    [?passenger].[?Name]
    /0/passenger/0/1/name/1/length:get_length


Selectors are usually applied to the long sequences of data which may not be reproducible. It is therefore wise to cover your featurization with try-except block and store the exception on the hard drive, so you could later build a test case with `original_object`.

`get_data_path()` returns the string representation of path inside data where the error has occured: somewhere around `obj['passenger']['Name']`. Symbol `?` means that these fields are optional, and _all fields_ are optional by default. If you want the selector that raises exception when the field does not exist, pass the `True` argument to the constructor of the `Selector`.

`get_code_path()` describes where the error occured within the hierarchy of selectors. By looking at this string, we can easily figure out that error occured somewhere around processing `name` with `get_length` method. If the deeper analysis is required, we may use the `representation` object we have previously built:


```python
representation[0]['passenger'][0]
```




    {0: {0: '[?passenger]'},
     1: {'name': {0: '[?Name]',
       1: {'length': '<function get_length at 0x7fd80a3c2040>',
        'title': '<function get_title at 0x7fd80a428ca0>'}},
      'Sex': {0: '[?Sex]'},
      'Age': {0: '[?Age]'}}}



Here we input the beginning of `get_code_path()` output and see the closer surroundings of the error.

## List featurization

In this section we will consider building features for a list of objects. This use case is rather rare, the examples is, for instance, building the features for a customer, basing on the articles he have purchased in the past.

In the Titanic setup, imagine that:
1. Our task is actually produce feature for cabins, not for passengers.
2. Our DOF is also flow of cabins, not passengers.

So one object looks like this:


```python
cabin_obj = source.get_data().where(lambda z: z['trip']['Cabin']=='F2').to_list()
cabin_obj
```




    [{'id': 149,
      'ticket': {'ticket.id': '230080', 'fare': 26.0, 'Pclass': 2},
      'passenger': {'Name': 'Navratil, Mr. Michel ("Louis M Hoffman")',
       'Sex': 'male',
       'Age': 36.5},
      'trip': {'Survived': 0,
       'SibSp': 0,
       'Patch': 2,
       'Cabin': 'F2',
       'Embarked': 'S'}},
     {'id': 194,
      'ticket': {'ticket.id': '230080', 'fare': 26.0, 'Pclass': 2},
      'passenger': {'Name': 'Navratil, Master. Michel M',
       'Sex': 'male',
       'Age': 3.0},
      'trip': {'Survived': 1,
       'SibSp': 1,
       'Patch': 1,
       'Cabin': 'F2',
       'Embarked': 'S'}},
     {'id': 341,
      'ticket': {'ticket.id': '230080', 'fare': 26.0, 'Pclass': 2},
      'passenger': {'Name': 'Navratil, Master. Edmond Roger',
       'Sex': 'male',
       'Age': 2.0},
      'trip': {'Survived': 1,
       'SibSp': 1,
       'Patch': 1,
       'Cabin': 'F2',
       'Embarked': 'S'}}]



We want to build the following features for this `cabin_obj`: the average fare and age of the passengers. 

So, to build such aggregated selectors, following practice is recommended:
1. build a `Selector` that selects the fields, and apply it to the list, building list of dictionaries
2. convert list of dictionaries into dictionary of lists
3. apply averager to each list.

Let's first do it step-by-step. `Listwise` applies arbitrary function (e.g., your selector) to the elements of the list:


```python
from tg.common.datasets.selectors import Listwise, Dictwise, transpose_list_of_dicts_to_dict_of_lists

cabin_features_selector = (Selector()
                           .select('passenger.Age','ticket.fare')
                          )
list_of_dicts = Listwise(cabin_features_selector)(cabin_obj)
list_of_dicts
```




    [{'Age': 36.5, 'fare': 26.0},
     {'Age': 3.0, 'fare': 26.0},
     {'Age': 2.0, 'fare': 26.0}]



`transpose_list_of_dicts_to_dict_of_lists` makes the "transposition" of list of dicts into dict of lists.


```python
dict_of_lists = transpose_list_of_dicts_to_dict_of_lists(list_of_dicts)
dict_of_lists
```




    {'Age': [36.5, 3.0, 2.0], 'fare': [26.0, 26.0, 26.0]}



Finally, `Dictwise` applies function to the elements of dictionary


```python
import numpy as np

Dictwise(np.mean)(dict_of_lists)
```




    {'Age': 13.833333333333334, 'fare': 26.0}



If you need a more complicated logic, such as applying different functions to different fields, you will need to extend `Dictwise` class.

All we have to do now is to assemble it to the pipeline. Since in our use cases we have used this pipeline several times, it's standardized in the following class:


```python
from tg.common.datasets.selectors import ListFeaturizer

selector = ListFeaturizer(cabin_features_selector, np.mean)
selector(cabin_obj)
```




    {'Age': 13.833333333333334, 'fare': 26.0}



### Quick dataset creations

The combination of `DataSource` and `Featurizer` allows you to quickly build the tidy dataset:


```python
source.get_data().take(3).select(titanic_selector).to_dataframe()
```

    2022-06-27 19:28:38.853372+00:00 WARNING: Missing field in FieldGetter
    2022-06-27 19:28:38.854069+00:00 WARNING: Missing field in FieldGetter
    2022-06-27 19:28:38.854707+00:00 WARNING: Missing field in FieldGetter





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>passenger_name_length</th>
      <th>passenger_name_title</th>
      <th>passenger_Sex</th>
      <th>passenger_Age</th>
      <th>ticket_id</th>
      <th>ticket_fare</th>
      <th>ticket_PClass</th>
      <th>trip_id</th>
      <th>trip_Survived</th>
      <th>trip_Cabin</th>
      <th>trip_Embarked</th>
      <th>trip_SibSp</th>
      <th>trip_Patch</th>
      <th>trip_Relatives</th>
      <th>processed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>Mr</td>
      <td>male</td>
      <td>22.0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>None</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2022-06-27 21:28:38.853902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51</td>
      <td>Mrs</td>
      <td>female</td>
      <td>38.0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>None</td>
      <td>2</td>
      <td>1</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2022-06-27 21:28:38.854524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>Miss</td>
      <td>female</td>
      <td>26.0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>None</td>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2022-06-27 21:28:38.855246</td>
    </tr>
  </tbody>
</table>
</div>



If your selector is small, you may also define `Selector` on the fly:


```python
(source
 .get_data()
 .take(3)
 .select(Selector().select('passenger.Name','trip.Survived'))
 .to_dataframe()
)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Summary

In this demo, we have presented how you may use `Selector` and other classes to convert complex, hierarchical objects into rows in the tidy dataset. 




# 2.3. Featurization Jobs and Datasets (tg.common.datasets.featurization)

In the `tg.common.datasets.access` and `tg.common.datasets.selectors`, we covered `DataSource` and `Selector` classes to build a tidy dataset from the external data source. For small datasets that can be build in several minutes, these two components are fine and you don't need anything more. However, for the bigger datasets, additional questions arise, like:

* Sometimes the data set is too large to hold in memory. Since selector produces rows one by one, it's not a big problem: we can just separate them into several smaller dataframes
* Sometimes we want to exclude some records from the dataset, or produce several rows per one object
* Sometimes we actually do not want the resulting dataframe, but some aggregated statistics instead
* And finally, sometimes we do not want to execute this procedure on our local machine. Instead, we want to deliver it to the cloud.

`tg.common.datasets.featurization` addresses these questions, offering `FeaturizationJob` and `UpdatableFeaturizationJob`, which are production-ready classes to create datasets on scale: 

* `FeaturizationJob` creates the whole dataset and cannot update its rows.
* `UpdatableFeaturizationJob` creates the dataset or updates it, if only some rows are changed.



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

    2022-06-27 19:28:42.858650+00:00 INFO: Featurization Job job at version v1 has started
    2022-06-27 19:28:42.861149+00:00 INFO: Fetching data
    2022-06-27 19:28:42.961584+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:28:43.025426+00:00 INFO: Uploading data
    2022-06-27 19:28:43.026883+00:00 INFO: Featurization job completed


Some notes: 

* `DataFrameFeaturizer`: When used in this way, it just applies `row_selector` to each data object from `source` and collects the results into pandas dataframes
* If no `location` is provided, the folder will be created automatically in the `Loc.temp_path` folder. Usually we don't care where the intermediate files are stored, as syncer takes care of them automatically.
* `MemoryFileSyncer`. The job creates files locally (in the `location` folder), and the uploads them to the remote destination. For demonstration purposes, we will "upload" data in the memory. `tg.common` also contains `S3FileSyncer` that syncs the files with `S3`. Interfaces for other storages may be written, deriving from `FileSyncer`. Essentialy, the meaning of `FileSyncer` is a connection between a specific location on the local disk and the location somewhere else. When calling `upload` or `download` methods, the class assures the same content of given files/folders.


The resulting files can be viewed in the following way:


```python
list(mem.cache)
```




    ['passengers/70a1cf74-68a2-4fc2-a8cc-6820901dcd35.parquet']



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

    2022-06-27 19:28:43.077634+00:00 INFO: Featurization Job job at version v1 has started
    2022-06-27 19:28:43.078705+00:00 INFO: Fetching data
    2022-06-27 19:28:43.157156+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:28:43.163353+00:00 INFO: Uploading data
    2022-06-27 19:28:43.164478+00:00 INFO: Featurization job completed





    ['passengers/2cef2bf5-ef73-485e-8939-f3eb54873832.parquet',
     'passengers/7456ec6e-40e8-4bf1-b96a-5fccc68e619c.parquet',
     'passengers/a9e67a6d-c435-41b1-bfff-1a13c8d4fc63.parquet',
     'passengers/2784951e-6392-414e-8a90-d02e013696cf.parquet']




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

    2022-06-27 19:28:43.190870+00:00 INFO: Featurization Job job at version v1 has started
    2022-06-27 19:28:43.193145+00:00 INFO: Fetching data
    2022-06-27 19:28:43.251880+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:28:43.262021+00:00 INFO: Uploading data
    2022-06-27 19:28:43.262974+00:00 INFO: Featurization job completed





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

    2022-06-27 19:28:43.289208+00:00 INFO: Featurization Job job at version v1 has started
    2022-06-27 19:28:43.291136+00:00 INFO: Fetching data
    2022-06-27 19:28:43.347860+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:28:43.363004+00:00 INFO: Uploading data
    2022-06-27 19:28:43.364114+00:00 INFO: Featurization job completed





    ['cabins/fde2b5e9-3cfa-4685-9da5-a0be9fadbfb6.parquet',
     'passengers/1d3160ef-40e9-43d6-af11-e0c0ba789413.parquet']



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






# 3. Machine Learning (tg.common.ml)

Machine learning is supported by the following modules:

* `tg.common.ml.dft`: a handy decorator over `sklearn` that applies transformers to the dataframes and keeps the names of the columns. Greatly improves debugging of the machine learning pipelines and simplifies data cleaning.
* `tg.common.ml.single_frame`: pipeline that applies simpler ML-algorithms (logistic regression, XGBoost) to the data. The requirement is that the data must fit to the memory after transformation all at once.
* `tg.common.ml.batched_training`: pipeline that applies more complex algorithms, typically neural networks, to the data that do not fit in the memory all at once after transformation. 




# 3.1. Data Cleaning (tg.common.ml.dft)

## Overview

The Data Cleaning phase occurs after Featurization. At this stage, we have the data as the tidy dataframe, but there are still:

* missing continuous values
* categorical values in need of transformation

In `sklearn` there are plenty of useful classes to address these problems. The only problem with them is that they do not keep the data in `pd.DataFrame` format, converting them to `numpy` arrays, thus losing the column names and making the debugging much harder.

`tg.common.ml.dft` (we will refer to it as `dft` for shortness) module offers a solution to this problem, wrapping the `sklearn` functionality and ensuring that the column names are preserved. 



## How to do it quickly and painlessly

This demo is mostly describing how `dft` is working and how to customize it. However, in our practice, we have found a perfect setup of data cleaning that we don't really customize, and we believe that this setup may be useful for other projects as well. 

So we will start with this quick solution, and then describe in details how it works. If the customization of data cleaning is not required, you may skip all the following parts of this demo.


```python
import pandas as pd
df = pd.read_csv('titanic.csv')
df = df.set_index('PassengerId')
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>5</th>
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




```python
from tg.common.ml import dft

tfac = dft.DataFrameTransformerFactory.default_factory(
    features = ['Pclass', 'Sex', 'Age', 'Cabin','Embarked'],
    max_values_per_category = 5
    )

tfac.fit_transform(df).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Age_missing</th>
      <th>Pclass_3</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Sex_male</th>
      <th>Sex_female</th>
      <th>Cabin_C23 C25 C27</th>
      <th>Cabin_G6</th>
      <th>Cabin_B96 B98</th>
      <th>Cabin_OTHER</th>
      <th>Cabin_NULL</th>
      <th>Embarked_S</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_NULL</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.530377</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.571831</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.254825</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.365167</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.365167</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



`tfac` is a data transformer in the sense of `sklearn`, it has the `fit`, `transform` and `fit_transform` method.

The default solution:
  * automatically determines if the feature is continuous or categorical
  * performs normalisation and imputation to continous variables, as well as adds the missing indicator
  * applies one-hot encoding to categorical variables, checking for None values, and also limits the amount of columns per feature, placing least-popular values in `OTHER` column.


## Class structure

With `dft`, you define a single transformer, which is a normal `sklearn` transformer with `fit`, `fit_transform` and `transform` methods. This is a composite transformer that has the following structure:

```
DataFrameTransformer
  ↳ (has) List[DataFrameColumnsTransformer]
               ↳ (is) ContinousTransformer
                      ↳ (has) scaler
                              ↳ (is) sklearn.preprocessing.StandardScaler, etc
                      ↳ (has) 
                              ↳ (is) sklearn.preprocessing.SimpleImputer, etc
                      ↳ (has) missing_indicator
                              ↳ (is) sklearn.impute.MissingIndicator
                              ↳ (is) dft.MissingIndicatorWithReporting
               ↳ (is) CategoricalTransformer2
               ↳ (is) CategoricalTransformer (obsolete version)
                      ↳ (has) replacement_strategy
                              ↳ (is) MostPopularStrategy
                              ↳ (is) TopKPopularStrategy
                      ↳ (has) postprocessor
                              ↳ (is) OneHotEncoderForDataFrame
```

Since such data structures are quite cumbersome to write and read, `dft` also contains a `DataFrameTransformerFactory` class which can be used in the most widespread scenarios to specify `DataFrameTransformer` quickly. 

**Note**: Regarding categorical features, two versions are available, `CategoricalTransformer` and `CategoricalTransformer2`. The latter is recommended: while `CategoricalTransformer` is much more flexible, this flexibility is almost never used in practice, and `CategoricalTransformer2` is significantly faster and memory-efficient.


We will now demonstrate all these classes in details. First, let's take a look at our dataset again:


```python
import pandas as pd
df = pd.read_csv('titanic.csv')
df = df.set_index('PassengerId')
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>5</th>
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



`Survived` is a label and should not go through the cleaning. 

`Pclass`, `SibSp`, `Parch` are integers, but in fact they are continous variables and we will convert them to the appropriate type.


```python
for c in ['Pclass','SibSp','Parch']:
    df[c] = df[c].astype(float)
```

## Continuous transformation

The dataset contains the following continuous columns:


```python
continuous_features = list(df.dtypes.loc[df.dtypes=='float64'].index)
continuous_features
```




    ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']




```python
df[continuous_features].describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



All features require normalization, and `Age` requires imputation. Since this is a quite standard case, the default instance of `ContinousTransformer` will do:


```python
tr = dft.DataFrameTransformer([
    dft.ContinousTransformer(
        columns=continuous_features
    )
])
tdf = tr.fit_transform(df)
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
    </tr>
    <tr>
      <th>PassengerId</th>
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
    </tr>
    <tr>
      <th>2</th>
      <td>-1.566107</td>
      <td>0.571831</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.786845</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.827377</td>
      <td>-0.254825</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.488854</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.566107</td>
      <td>0.365167</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.420730</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.827377</td>
      <td>0.365167</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.486337</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



We see new column, `Age_missing`, which is `True` is the `Age` value was missing for this row. For other columns, we don't see this, this is the default behaviour of the `MissingIndicator`

Let's check the distribution of `Age`:


```python
tdf.Age.hist()
pass
```


    
![png](README_images/tg.common.ml.dft_output_17_0.png?raw=true)
    


By the range of values it's easy to see that the `StandardScaler` was used. We can of course replace the scaler, as well as other components of the transformer


```python
from sklearn.preprocessing import MinMaxScaler

tr = dft.DataFrameTransformer([
    dft.ContinousTransformer(
        columns=continuous_features,
        scaler = MinMaxScaler(feature_range=(-1,1))
    )
])
tr.fit_transform(df).Age.hist()
pass
```


    
![png](README_images/tg.common.ml.dft_output_19_0.png?raw=true)
    


Some notes on missing indicator. When the sklearn Missing indicator is used, the error is thrown when the column that did not happen to be None in training, does so in test:


```python
import traceback

test_df = pd.DataFrame([dict(Survived=0, Age=30, SibSp=0, Fare=None)]).astype(float)
from sklearn.impute import MissingIndicator

tr = dft.DataFrameTransformer([
    dft.ContinousTransformer(
        columns=continuous_features,
        missing_indicator = MissingIndicator()
    )
])
tr.fit(df)
try:
    tr.transform(test_df)
except ValueError as exp:
    traceback.print_exc() #We catch the exception so the Notebook could proceed uninterrupted
```

    2022-06-27 19:28:53.100086+00:00 WARNING: Missing column in ContinuousTransformer
    2022-06-27 19:28:53.103057+00:00 WARNING: Missing column in ContinuousTransformer


    Traceback (most recent call last):
      File "/tmp/ipykernel_31087/2155061853.py", line 14, in <module>
        tr.transform(test_df)
      File "/home/yura/Desktop/repos/tg/tg/common/ml/dft/architecture.py", line 49, in transform
        for res in transformer.transform(df):
      File "/home/yura/Desktop/repos/tg/tg/common/ml/dft/column_transformers.py", line 92, in transform
        missing = self.missing_indicator.transform(subdf)
      File "/home/yura/anaconda3/envs/tg/lib/python3.8/site-packages/sklearn/impute/_base.py", line 885, in transform
        raise ValueError(
    ValueError: The features [4] have missing values in transform but have no missing values in fit.


This effect can be quite annoying if the column has `None` value in exceptionally low amount of fringe cases, so when doing random train/test split this value may become absent. To avoid that, `dft` improves the `sklearn` class:


```python
tr = dft.DataFrameTransformer([
    dft.ContinousTransformer(
        columns=continuous_features,
        missing_indicator = dft.MissingIndicatorWithReporting() # This class is used by default
    )
])
tr.fit(df)
tr.transform(test_df)
```

    2022-06-27 19:28:53.133821+00:00 WARNING: Missing column in ContinuousTransformer
    2022-06-27 19:28:53.135301+00:00 WARNING: Missing column in ContinuousTransformer
    2022-06-27 19:28:53.140779+00:00 WARNING: Unexpected None in MissingIndicatorWithReporting
    2022-06-27 19:28:53.142468+00:00 WARNING: Unexpected None in MissingIndicatorWithReporting
    2022-06-27 19:28:53.143747+00:00 WARNING: Unexpected None in MissingIndicatorWithReporting





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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.020727</td>
      <td>-0.474545</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, the exception is replaced with a warning in the `Logger` instead. They keys to the message carry the information about the affected entities:


```python
from tg.common import Logger

Logger.initialize_kibana()

tr.transform(test_df)
```

    {"@timestamp": "2022-06-27 19:28:53.160523+00:00", "message": "Missing column in ContinuousTransformer", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/tg/tg/common/ml/dft/column_transformers.py", "path_line": 77, "column": "Pclass"}
    {"@timestamp": "2022-06-27 19:28:53.164905+00:00", "message": "Missing column in ContinuousTransformer", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/tg/tg/common/ml/dft/column_transformers.py", "path_line": 77, "column": "Parch"}
    {"@timestamp": "2022-06-27 19:28:53.169612+00:00", "message": "Unexpected None in MissingIndicatorWithReporting", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/tg/tg/common/ml/dft/miscellaneous.py", "path_line": 36, "column": "Pclass"}
    {"@timestamp": "2022-06-27 19:28:53.170205+00:00", "message": "Unexpected None in MissingIndicatorWithReporting", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/tg/tg/common/ml/dft/miscellaneous.py", "path_line": 36, "column": "Parch"}
    {"@timestamp": "2022-06-27 19:28:53.170654+00:00", "message": "Unexpected None in MissingIndicatorWithReporting", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/tg/tg/common/ml/dft/miscellaneous.py", "path_line": 36, "column": "Fare"}





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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.020727</td>
      <td>-0.474545</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



The recommendation is:

* If these None values are potentially critical for the model correctness, specify  `sklearn.impute.MissingIndicator`
* If they aren't, don't specify, so the default `dft.MissingIndicatorWithReporting` will be used. Check output warnings to monitor the issue.

## Categorical values

Categorical variables are processed by `CategoricalTransformer` with the following routine:

* Replace all the values with their string representation (for the sake of type consistancy)
* Also replace None with a provided string constant, `'NONE'` by default. After that, None is treated like normal value of categorical variable
* Apply replacement strategy: e.g. only keep N most popular values and ignore all else.
* Apply post-processing, e.g. One-Hot encoding

These are categorical variables of out dataset:


```python
categorical_variables = list(df.dtypes.loc[df.dtypes!='float64'].index)
categorical_variables
```




    ['Survived', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']




```python
df[categorical_variables].head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Ticket</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>A/5 21171</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>PC 17599</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>STON/O2. 3101282</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>113803</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>373450</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.Series({c: len(df[c].unique()) for c in categorical_variables}).sort_values()
```




    Survived      2
    Sex           2
    Embarked      4
    Cabin       148
    Ticket      681
    Name        891
    dtype: int64



I will exclude Ticket and Name from features, because they are near to unique for each row so it does not make sense to include them. I will also exclude `Survived` because it is a label.


```python
for c in ['Ticket','Name','Survived']:
    categorical_variables.remove(c)
```


```python
df[categorical_variables].isnull().sum(axis=0)
```




    Sex           0
    Cabin       687
    Embarked      2
    dtype: int64



Categorical column transformer essentially does the following:

1. Converts all the values to string format. None/NaN is converted to a `NONE` string (parametrized in constructor)
2. Somehow deals with values: removes excessive values or values unseen during the training by replacing it with something.
3. Postprocesses the result with e.g. one-hot encoding or converting to indices if required by the model.


```python
tr = dft.DataFrameTransformer([
    dft.CategoricalTransformer(
        columns=categorical_variables
    )
])
tdf = tr.fit_transform(df)
tdf.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>NONE</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>NONE</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>female</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>male</td>
      <td>NONE</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Imagine we performed a train/test split so that `Embarked` was always not null in the training set.


```python
tr = dft.DataFrameTransformer([
    dft.CategoricalTransformer(
        columns=['Embarked']
    )
])
tr.fit(df.loc[~df.Embarked.isnull()])
tr.transform(df.loc[df.Embarked.isnull()])
```

    {"@timestamp": "2022-06-27 19:28:53.270541+00:00", "message": "Unexpected value in MostPopularStrategy", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/tg/tg/common/ml/dft/column_transformers.py", "path_line": 124, "column": "Embarked", "value": "NONE"}





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62</th>
      <td>S</td>
    </tr>
    <tr>
      <th>830</th>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



We see it was converted to `S`. This is because by default, `MostPopular` strategy is used, and this strategy replaces the unseen values with the most popular ones. `S` is way more popular than others.


```python
df.groupby(df.Embarked).size()
```




    Embarked
    C    168
    Q     77
    S    644
    dtype: int64



For `Cabin` field, however, this strategy doesn't make much sense, because the number of different categories is too high. In one-hot encoding case, the number of columns will be enormous. So we can use `TopKPopularStrategy` in this case


```python
tr = dft.DataFrameTransformer([
    dft.CategoricalTransformer(
        columns=['Cabin'],
        replacement_strategy= dft.TopKPopularStrategy(10,'OTHER')
    )
])
tdf = tr.fit_transform(df)
tdf.groupby('Cabin').size().sort_values(ascending=False)
```




    Cabin
    NONE           687
    OTHER          175
    B96 B98          4
    C23 C25 C27      4
    G6               4
    C22 C26          3
    D                3
    E101             3
    F2               3
    F33              3
    C65              2
    dtype: int64



So we have 11 values, which is a constructor parameter 10 + 1 value for `'OTHER'` (you may replace `'OTHER'` with an arbitrary string constant in constructor). The top-popular category is `'None'`. Still, there are some cabins shared across several passenger, and that might allow us to predict the fate of other passengers in this cabings correctly - but still keeping this in control by limiting the amount of cabins.

After applying replacement strategy, we will _never_ have the values that we didn't have in a training set. This is crucial for sucessful run of the machine learning algorithm that are located down the stream.

All the messages about unexpected values are stored in `TgWarningStorage`

Finally, we can implement one-hot encoding, or other postprocessing, required for many models.


```python
from sklearn.preprocessing import OneHotEncoder

tr = dft.DataFrameTransformer([
    dft.CategoricalTransformer(
        columns=['Embarked','Cabin'],
        postprocessor=dft.OneHotEncoderForDataframe()
    )
])
tr.fit_transform(df).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked_C</th>
      <th>Embarked_NONE</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Cabin_A10</th>
      <th>Cabin_A14</th>
      <th>Cabin_A16</th>
      <th>Cabin_A19</th>
      <th>Cabin_A20</th>
      <th>Cabin_A23</th>
      <th>...</th>
      <th>Cabin_F E69</th>
      <th>Cabin_F G63</th>
      <th>Cabin_F G73</th>
      <th>Cabin_F2</th>
      <th>Cabin_F33</th>
      <th>Cabin_F38</th>
      <th>Cabin_F4</th>
      <th>Cabin_G6</th>
      <th>Cabin_NONE</th>
      <th>Cabin_T</th>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 152 columns</p>
</div>



## `CategoricalTransformer2`

In our practice, we always used `CategoricalTransformer` with `TopKPopularStrategy` and `OneHotEncoderForDataframe`, so we didn't really make any use of the class's flexibility. However, this flexibility adds overheads, and significantly reduces the performance of the class. To address this issue, we developed `CategoricalTransformer2`, which does the very same transformation as `CategoricalTransformer` with `TopKPopularStrategy` and `OneHotEncoderForDataframe`, but in a very efficient way.


```python
tr = dft.DataFrameTransformer([
    dft.CategoricalTransformer2(
        columns=['Embarked','Cabin'],
        max_values=4
    )
])
tr.fit_transform(df).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked_S</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_NULL</th>
      <th>Cabin_C23 C25 C27</th>
      <th>Cabin_G6</th>
      <th>Cabin_OTHER</th>
      <th>Cabin_NULL</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Combining transformers

You can combine different transformers for different columns.

The code below basically creates the transformer to feed Titanic Dataset into Logistic Regression


```python
tr = dft.DataFrameTransformer([
    dft.ContinousTransformer(
        columns=continuous_features
    ),
    dft.CategoricalTransformer(
        columns= ['Sex','Embarked'],
        postprocessor=dft.OneHotEncoderForDataframe()
    ),
    dft.CategoricalTransformer2(
        columns=['Cabin'],
        max_values=5
    )
])
tr.fit_transform(df).head()
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
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>Embarked_C</th>
      <th>Embarked_NONE</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Cabin_C23 C25 C27</th>
      <th>Cabin_G6</th>
      <th>Cabin_B96 B98</th>
      <th>Cabin_OTHER</th>
      <th>Cabin_NULL</th>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.566107</td>
      <td>0.571831</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.786845</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.827377</td>
      <td>-0.254825</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.488854</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.827377</td>
      <td>0.365167</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.486337</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Transformer factories

There are two problems with the transformers. The first is subjective: the initialization code for a large transformer sets is ugly. The second is more problematic: in order to build the transformers, you usually need to see the data first: for instance, to decide, which categorical columns should be processed with default replacement strategy, and which should be processed with `TopKReplacementStrategy`.

Both of these problems are solved by DataFrameTransformerFactory. This is the class is an `sklearn` transformer. When `fit` is called, it creates the DataFrameTransformer according to its setting, and fits this transformer. When transforming, it just passes the data to the transformer that should be created ealier.

The following code is creating the factory for the Titanic dataset:



```python
from functools import partial

tfac = (dft.DataFrameTransformerFactory()
 .with_feature_block_list(['Survived','Name','Ticket'])
 .on_continuous(dft.ContinousTransformer)
 .on_categorical_2()
 .on_rich_category(10, partial(
     dft.CategoricalTransformer, 
     postprocessor=dft.OneHotEncoderForDataframe(), 
     replacement_strategy = dft.TopKPopularStrategy(10,'OTHER')
)))

tfac.fit_transform(df).head()
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
      <th>Cabin_C22 C26</th>
      <th>Cabin_E101</th>
      <th>Cabin_F33</th>
      <th>Cabin_D</th>
      <th>Cabin_OTHER</th>
      <th>Cabin_NULL</th>
      <th>Embarked_S</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_NULL</th>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



`on_categorical_2` and `on_categorical` are different in the same way as `CategoricalTransformer2` and `CategoricalTransformer`. So, using `on_categorical_2` is advised.

`partial` is used here for the following reason. `on_continuous`, `on_categorical` etc methods receive the function, that accepts the list of column names and creates a `DataFrameColumnTransformer` for them. Normally, we would write something like:

```
on_categorical(
    lambda features: dft.CategoricalTransformer(features, postprocessor=dft.OneHotEncoderForDataframe())
```

But the `DataFrameTransformerFactory` object is typically to be delivered to the remote server, and for this it has to be serializable. Unfortunately, lambdas is Python are not serializable. Therefore, we need to replace them, and `partial` is a good tool.

`dft.DataFrameTransformerFactory` also has methods `default_factory` that essentially return the code block above. Note that we don't have `default_factory_2` method, as there is only one "default" solution, and it works with `CategoricalTransformer2`.



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


    2022-06-27 19:28:56.739258+00:00 INFO: Starting stage 1/1
    2022-06-27 19:28:56.799082+00:00 INFO: Completed stage 1/1


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
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7f2b378cadf0>),
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
      <td>0.862917</td>
      <td>0.998319</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.826485</td>
      <td>0.998715</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.851331</td>
      <td>0.997248</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.815813</td>
      <td>0.999520</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.887391</td>
      <td>0.997797</td>
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
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7f2b34097bb0>),
                    ('ColumnNamesKeeperAfterTransformation', ColumnNamesKeeper()),
                    ('Model',
                     Pipeline(steps=[('CategoricalVariablesSetter',
                                      <tg.common.ml.single_frame_training.model_provider.CatBoostWrap object at 0x7f2b34088280>),
                                     ('Model',
                                      <catboost.core.CatBoostClassifier object at 0x7f2b3826e8b0>)]))])



`model_fix` is a function, that updates the model to something else. In our case, the initial instance of catboost model `Model` was replaced with a Pipeline, containing two steps. The second step is `Model`. The first step is a wrapper, that accepts the dataset, processed by transformers, understands which columns are categorical, and then sets the list of this columns to the `Model`

## Artificiers

_Artificier_ is an interface to inject an arbitrary code to the training process. So far, we had two use cases for artificiers:
* Remove model from the training result. The model may be huge and we may not be even interested in the model per se, just by it's metrics.
* Get the feature significance. Many algorithms allow us to extract feature significance from the model, which can be used in business analysis without the model itself.

Let's use write an artificier to discover the most important features in our dataset. 


```python
import tg.common.ml.single_frame_training as sft
from sklearn.metrics import roc_auc_score

class SignificanceArtificier:
    def run(self, model_info):
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


    
![png](README_images/tg.common.ml.single_frame_training_output_46_0.png?raw=true)
    


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
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7f2b192db580>),
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
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7f2b2049aca0>),
                    ('ColumnNamesKeeperAfterTransformation', ColumnNamesKeeper()),
                    ('Model', LogisticRegression())])



With Training Grounds, it is possible to perform hyperparameter optimization of single-frame model. If the model requires a significant time to train, we should use sagemaker and hyperopt. But sometimes it can be executed locally. For that, we offer `Kraken` class. 

Kraken does exactly one thing: it executes any given method over set of parameters, and brings the result into big pandas dataframe. Kraken supports exception handling as well as caching intermediate result on the disk for further restart, and this functionality is well-tested.


```python
from tg.common.ml.kraken import Kraken

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


    
![png](README_images/tg.common.ml.single_frame_training_output_55_0.png?raw=true)
    


We see the classic image for logistic regression  and random forest: while they perform comparable on the test set, the random forests fit extremely well on the train set.



# 3.3. Batched training (tg.common.ml.batched_training)

## Overview of batched training process

Batched training is used when the training data is synthetic and so huge, that it cannot fit into memory. Examples are:

* Customer-article fit model. The features of customers and articles can fit into memory, but when we combine them by instances of articles, that were sent to customers, the data explode.
* Natural language processing. When, for instance, processing the sentence of N words, we may wish to construct the training samples so that network predicts $W_i$ word from $(W_1,\ldots,W_{i-1})$ for $i\in[1, N]$. In this case, the data also explode.

Current version of batched training process is *not* dealing with the situation, when the initial data themselves are too big to fit the memory. If this is the case, the current strategy is to get a bigger AWS instance. If it's not feasible, the approach can theoretically be adopted for this case as well.

In batched training, the data are synthesized in _batches_, one after another, and therefore memory is not overused. Consequently, the model must support the iterative training. In our use cases this is always pytorch network, but of course the architecture is not limited to neural networks or some particular implementation of them.

## Data Bundles

As a dataframe is the "all-inclusive" object containing the data for single-frame training, _data bundle_ is such for batched training. The bundle is a set of dataframes, that contains all the data required for batches synthesis.

You normally start batched training by creating such bundle, and we will do the same for the Titanic dataset. We will do it in an overcomplicated way, that is totally unnessesary for this particular task, but will allow us to demonstrate available batching techniques. We will separate the available information in Titanic dataset into two parts: information about ticket and information about customer.


```python
import pandas as pd
df = pd.read_csv('titanic.csv')
df = df.set_index('PassengerId')
for c in ['Pclass','SibSp','Parch','Survived']:
    df[c] = df[c].astype(float)
```

In rare cases Ticket number does not fully determine the Fare or Embarked column, but we will ignore the exceptions.


```python
tdf = df[['Ticket','Pclass','Fare','Embarked']].drop_duplicates()
bad_tickets = tdf.groupby('Ticket').size()
bad_tickets = bad_tickets.loc[bad_tickets>1]
tdf.loc[tdf.Ticket.isin(bad_tickets.index)].sort_values('Ticket')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticket</th>
      <th>Pclass</th>
      <th>Fare</th>
      <th>Embarked</th>
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
      <th>271</th>
      <td>113798</td>
      <td>1.0</td>
      <td>31.0000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>843</th>
      <td>113798</td>
      <td>1.0</td>
      <td>31.0000</td>
      <td>C</td>
    </tr>
    <tr>
      <th>139</th>
      <td>7534</td>
      <td>3.0</td>
      <td>9.2167</td>
      <td>S</td>
    </tr>
    <tr>
      <th>877</th>
      <td>7534</td>
      <td>3.0</td>
      <td>9.8458</td>
      <td>S</td>
    </tr>
    <tr>
      <th>270</th>
      <td>PC 17760</td>
      <td>1.0</td>
      <td>135.6333</td>
      <td>S</td>
    </tr>
    <tr>
      <th>326</th>
      <td>PC 17760</td>
      <td>1.0</td>
      <td>135.6333</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



So the ticket dataframe is:


```python
ticket_df = df[['Ticket','Pclass','Fare','Embarked']].drop_duplicates().drop_duplicates('Ticket').set_index('Ticket')
ticket_df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>Ticket</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A/5 21171</th>
      <td>3.0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>PC 17599</th>
      <td>1.0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>STON/O2. 3101282</th>
      <td>3.0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>113803</th>
      <td>1.0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>373450</th>
      <td>3.0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Name is unique across passengers, so we will use it as an index.


```python
df.groupby('Name').size().sort_values(ascending=False).head()
```




    Name
    Abbing, Mr. Anthony             1
    Nysveen, Mr. Johan Hansen       1
    Nicholson, Mr. Arthur Ernest    1
    Nicola-Yarred, Master. Elias    1
    Nicola-Yarred, Miss. Jamila     1
    dtype: int64




```python
passenger_df = df[['Name','Sex','Age','SibSp','Parch']].set_index('Name')
passenger_df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Braund, Mr. Owen Harris</th>
      <td>male</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</th>
      <td>female</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Heikkinen, Miss. Laina</th>
      <td>female</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Futrelle, Mrs. Jacques Heath (Lily May Peel)</th>
      <td>female</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Allen, Mr. William Henry</th>
      <td>male</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Now, we will build the frame that sums up the information:


```python
index_df = df[['Name','Ticket','Cabin','Survived']].copy()
index_df.Survived = index_df.Survived.astype(float)
index_df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Survived</th>
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
      <td>Braund, Mr. Owen Harris</td>
      <td>A/5 21171</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>PC 17599</td>
      <td>C85</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Heikkinen, Miss. Laina</td>
      <td>STON/O2. 3101282</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>113803</td>
      <td>C123</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Allen, Mr. William Henry</td>
      <td>373450</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from tg.common.ml import batched_training as bt

bundle = bt.DataBundle(
    index = index_df,
    passengers = passenger_df,
    tickets = ticket_df
)
```

Of course, such complicated structure of this bundle is totally unnesessary in this case, and we could just use the entire original dataframe as an index frame. 

Still, it shows how to bundle data together: we use different data sources (e.g. different table from SQL) without any joining or transformation, and bind them together by index frame.

Let's also save this bundle for later use:


```python
bundle.save('temp/bundle')
```

### Indexed Data Bundle

We see that one of the dataframes is "special" in the sense that it carries a top-level information that binds together other frames. This special role is reflected in `IndexedDataBundle` class, that contains `index_frame` and `bundle`. 


```python
ibundle = bt.IndexedDataBundle(bundle.index, bundle)
```

## Batcher

Batch is a dictionary with several dataframes that is used as a direct argument to the model. They have to be aligned (in our case, the tickets should correspond to the respective passengers), and transformed.

The pipeline to produce batches from `IndexedDataBundle` is:
1. Separate `index_frame` into parts. Each row of `index_frame` defines one individual training sample, so the batch is defined by a set of the rows, so to say the subset of `index_frame`. This part is done by `TrainingStrategy` class. 
2. For each part, pull the data from `data_frames` of the bundle and apply transformers to them. The resulting dataframes are then returned as a dictionary, which is then fed to the model for training or prediction. This is done by one or many `Extractor`.

The size of the batch is determined by the parameter `batch_size`. The rule of thumb is to select the biggest possible `batch_size` that still does not overfill the memory of the machine.

*Tip:* to avoid confusion between extracting with `Extractors` and transforming with `dft`, remember simple rule: transformation is about _columns_, while extraction is about _rows_.

### Training Strategy

Simple batcher strategy simply separates the frame into subsequent parts of the desired size. Let's create a test dataframe we will use to demonstrate different strategies:


```python
test_df = pd.DataFrame(dict(x=list(range(5))))
test_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



`SimpleBatchingStrategy` simply separates the dataframe into the consequent subsets of appropriate size


```python
strategy = bt.SimpleBatcherStrategy()
batch_count = strategy.get_batch_count(3,test_df)
print(batch_count)
for i in range(batch_count):
    print(list(strategy.get_batch(3,test_df,i)))
```

    2
    [0, 1, 2]
    [3, 4]


We also have `PriorityRandomBatcherStrategy`, that randomly samples rows from index frame, paying attention to the indicated weights. That can be used to balance underrepresented samples in the dataset. All of the generated batches will be of the `batch_size`, even if it's greater that dataframe size.


```python
strategy = bt.PriorityRandomBatcherStrategy('x',deduplicate=False)
batch = strategy.get_batch(10000, test_df, 0)
test_df.loc[batch].groupby('x').size()
```




    x
    1    1016
    2    2025
    3    2975
    4    3984
    dtype: int64



The Titanic dataset's balance is okayish


```python
ibundle.index_frame.groupby('Survived').size()
```




    Survived
    0.0    549
    1.0    342
    dtype: int64



But let's use PriorityRandomBatcherStrategy nevertheless. First, we need to compute the priority of each row, which reflects imbalance. We have a function to do so:


```python
import copy

ibundle_fixed = copy.deepcopy(ibundle)
ibundle_fixed.index_frame['priority'] = bt.PriorityRandomBatcherStrategy.make_priorities_for_even_representation(
    df = ibundle_fixed.index_frame,
    column = 'Survived'
)
ibundle_fixed.index_frame.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Survived</th>
      <th>priority</th>
    </tr>
    <tr>
      <th>PassengerId</th>
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
      <td>Braund, Mr. Owen Harris</td>
      <td>A/5 21171</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.001821</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>PC 17599</td>
      <td>C85</td>
      <td>1.0</td>
      <td>0.002924</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Heikkinen, Miss. Laina</td>
      <td>STON/O2. 3101282</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.002924</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>113803</td>
      <td>C123</td>
      <td>1.0</td>
      <td>0.002924</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Allen, Mr. William Henry</td>
      <td>373450</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.001821</td>
    </tr>
  </tbody>
</table>
</div>



We see that survived passengers have slighly greater chances to make it to the batch.

**NOTE**: you probably wouldn't do it like that in the real training. The thing is, you don't want to complicate data bundle creation with a model-specific steps like this one. We only do it to demonstrate, how the final Batcher is going to perform. We will show how to do this step properly a little later


By the way, a good rule of thumb here is: **if you need data bundle when initializing the model, you're doing something wrong!** This is true for Single-Frame training as well, but it is especially true for batcher training, because even in our use cases some datasets are too big to be opened on our laptops, and you can only open them at the remote training instance.

### Extractors

Extractors look at the index, provided by `BatcherStrategy`, extract data from bundle and apply transformer. In this demo, as well as in many real application, it's done by joining the dataframes with the `index_frame`. This is implemented in `PlainExtractor` class.

Let's define a small sample to see how the results of the extractors look like:


```python
ibundle_sample = ibundle_fixed.change_index(ibundle_fixed.index_frame.iloc[10:13])
ibundle_sample.index_frame
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Survived</th>
      <th>priority</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>PP 9549</td>
      <td>G6</td>
      <td>1.0</td>
      <td>0.002924</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>113783</td>
      <td>C103</td>
      <td>1.0</td>
      <td>0.002924</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Saundercock, Mr. William Henry</td>
      <td>A/5. 2151</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.001821</td>
    </tr>
  </tbody>
</table>
</div>



The most simple case is when data extracted from index itself, which is the case for the label in our case.


```python
label_extractor = bt.PlainExtractor.build(name='labels').apply(take_columns='Survived')
label_extractor.extract(ibundle_sample)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Cabin also should be extracted from the index, but this time we will need a transformer.

**NOTE**: you don't need to `fit` anything manually! Again, we do it just to demonstrate how the classes work. All the fitting is done internally, you only need to define the instances of the corresponding classes. 


```python
from tg.common.ml import dft

tfac = dft.DataFrameTransformerFactory.default_factory

cabin_extractor = bt.PlainExtractor.build(name='cabin').index().apply(transformer=tfac(), take_columns='Cabin')
cabin_extractor.fit(ibundle_fixed)
cabin_extractor.extract(ibundle_sample)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cabin_C23 C25 C27</th>
      <th>Cabin_G6</th>
      <th>Cabin_B96 B98</th>
      <th>Cabin_F2</th>
      <th>Cabin_C22 C26</th>
      <th>Cabin_E101</th>
      <th>Cabin_F33</th>
      <th>Cabin_D</th>
      <th>Cabin_C78</th>
      <th>Cabin_B57 B59 B63 B66</th>
      <th>...</th>
      <th>Cabin_C123</th>
      <th>Cabin_C124</th>
      <th>Cabin_C125</th>
      <th>Cabin_C126</th>
      <th>Cabin_D36</th>
      <th>Cabin_C2</th>
      <th>Cabin_C83</th>
      <th>Cabin_E25</th>
      <th>Cabin_OTHER</th>
      <th>Cabin_NULL</th>
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
      <th>11</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 25 columns</p>
</div>



**Note** that `tfac` here is not an object, but a function that constructs the object. This is because we will have several extractors, and each of them have to have their own transformer! This is a common error and in fact `PlainExtractor` makes a deepcopy of transformer to circumvent the problem if one appears due to the oversight.

To extract features from other dataframes, we will need to add `join` method.


```python
passenger_extractor = (bt.PlainExtractor.build('passengers')
                       .index()
                       .join(frame_name='passengers', on_columns='Name')
                       .apply(transformer=tfac())
                      )
passenger_extractor.fit(ibundle_fixed)
passenger_extractor.extract(ibundle_sample)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Age_missing</th>
      <th>Sex_male</th>
      <th>Sex_female</th>
    </tr>
    <tr>
      <th>PassengerId</th>
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
      <th>11</th>
      <td>-1.770360</td>
      <td>0.432793</td>
      <td>0.767630</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.949591</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.668153</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Notice that the resulting dataframe is indexed the same way the `ibundle_sample.index_frame` is.

Let's repeat this for tickets as well.


```python
ticket_extractor = (bt.PlainExtractor.build(name='tickets')
                    .index()
                    .join(frame_name='tickets', on_columns='Ticket')
                    .apply(transformer=tfac())
                   )
ticket_extractor.fit(ibundle_fixed)
ticket_extractor.extract(ibundle_sample)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Fare</th>
      <th>Embarked_S</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_NULL</th>
    </tr>
    <tr>
      <th>PassengerId</th>
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
      <th>11</th>
      <td>0.827377</td>
      <td>-0.312156</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-1.566107</td>
      <td>-0.113831</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.827377</td>
      <td>-0.486320</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Now, we have 4 extractors, but we really do not need so much dataframes in batches. We only need have features and labels. So we will unite the feature extractors into one:


```python
feature_extractor = bt.CombinedExtractor('features',[cabin_extractor, passenger_extractor, ticket_extractor])
feature_extractor.fit(ibundle_fixed)
feature_extractor.extract(ibundle_sample)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cabin_Cabin_C23 C25 C27</th>
      <th>cabin_Cabin_G6</th>
      <th>cabin_Cabin_B96 B98</th>
      <th>cabin_Cabin_F2</th>
      <th>cabin_Cabin_C22 C26</th>
      <th>cabin_Cabin_E101</th>
      <th>cabin_Cabin_F33</th>
      <th>cabin_Cabin_D</th>
      <th>cabin_Cabin_C78</th>
      <th>cabin_Cabin_B57 B59 B63 B66</th>
      <th>...</th>
      <th>passengers_Parch</th>
      <th>passengers_Age_missing</th>
      <th>passengers_Sex_male</th>
      <th>passengers_Sex_female</th>
      <th>tickets_Pclass</th>
      <th>tickets_Fare</th>
      <th>tickets_Embarked_S</th>
      <th>tickets_Embarked_C</th>
      <th>tickets_Embarked_Q</th>
      <th>tickets_Embarked_NULL</th>
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
      <th>11</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.767630</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.827377</td>
      <td>-0.312156</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-0.473674</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.566107</td>
      <td>-0.113831</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-0.473674</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.827377</td>
      <td>-0.486320</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 37 columns</p>
</div>



### Batcher

Now we are ready to define the Batcher:


```python
batcher = bt.Batcher(
    batch_size = 100,
    extractors = [feature_extractor, label_extractor],
    batching_strategy = bt.PriorityRandomBatcherStrategy('priority')
)
```

Let's take a look at the batch produced


```python
batch = batcher.fit_extract(ibundle_fixed)
list(batch)
```




    ['index', 'features', 'labels']



The batch is balanced


```python
batch['labels'].groupby('Survived').size()
```




    Survived
    0.0    51
    1.0    47
    dtype: int64



**Note:** due to the technical reasons, `PlainExtractor`, as well as other extractors, do not support extraction in case when the output of `BatchingStrategy` contains duplicated rows. This is why by default, `PriorityRandomBatcherStrategy` deduplicates them, and therefore cannot return more rows that there are in the bundle. We consider fixing this issue in the future releases, but since the datasets are normaly (much) bigger than batches, not with the high priority.


```python
test_batcher = bt.Batcher(
    batch_size = 1000,
    extractors = [feature_extractor, label_extractor],
    batching_strategy = bt.PriorityRandomBatcherStrategy('priority')
)
test_batch = test_batcher.fit_extract(ibundle_fixed)
test_batch['labels'].groupby('Survived').size()
```




    Survived
    0.0    323
    1.0    254
    dtype: int64



### Minibatches

Batch size has the following effects on training:
  1. It defines amount of memory, consumed by the training instance
  1. It defines the velocity of training process, as smaller batches have bigger overheads (per record)
  1. It defines the quality of neural network training
  
[1] and [2] suggest we pick the biggest batch size that fits to the instance memory. However, the quality is often better with much smaller batches.

This is why minibatches are introduced. Once the heavy work of joining and transforming is done and the batch is computed, it can be further subdivided to _mini-batches_.


```python
mini_batch_indices = batcher.get_mini_batch_indices(mini_batch_size = 10, batch = batch)
mini_batch = batcher.get_mini_batch(index = mini_batch_indices[0], batch = batch)
mini_batch['index'].shape
```




    (10, 5)



## Model Handler

`ModelHandler` is another important component of the batched training. It handles the model, namely:

* Instantiates the model. For neural network, this step is non-trivial, because we don't really know how much features we have before we fit all the extractors. Thus, we don't really know how much inputs network should have. To instantiate the model, a sample batch is created and passed to the initialization method to address this.
* Implements training on one batch. Again, this is more complicated in case of neural networks than just calling a `fit` method. 
* Implements the prediction. For different tasks, we interpret the network's output differently. In Single Frame Training, we had Evaluation to address this. Still, here the options are more plentiful, so it's moved into `ModelHandler` as well.

In `tg.common.ml.batched_training.torch` there is a generic definition for such `ModelHandler` that we will cover in the corresponding demo. Here, we will define `ModelHandler` from scratch, to demonstrate its logic.



```python
import torch


class TorchNetwork(torch.nn.Module):
    def __init__(self, sizes):
        super(TorchNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            
    def forward(self, input):
        X = torch.tensor(input.astype(float).values).float()
        for layer in self.layers:
            X = layer(X)
            X = torch.sigmoid(X)
        return X
        
        
class TorchHandler(bt.BatchedModelHandler):
    def instantiate(self, task, input):
        sizes = [input['features'].shape[1], 10, 1]
        self.network = TorchNetwork(sizes)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.5)
        self.loss = torch.nn.MSELoss()

    def train(self, input):
        X, y = input['features'], input['labels']
        self.optimizer.zero_grad()
        output = self.network(X)
        targets = torch.tensor(y.values).float()
        loss = self.loss(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, input):
        X, y = input['features'], input['labels']
        output = self.network(X)
        output = output.flatten().tolist()
        df = pd.DataFrame(dict(predicted=output, true=y[y.columns[0]]))
        return df
    

```

## Training task

Aside from `ModelHandler` and `Batcher`, `BatcherTrainingTask` is initialized with the classes, already covered in the previous demo: 
* `splitter` to split the index frame to test and train dataset,
  * For batch training, the splitter _must_ return just one split. K-fold is not supported here, because the time of one model training is simply too long, and in this case we suggest the parallel training.
    * The use case for multiple splits _could_ be a model, than trains on first N months and predicts N+1-th, then updates on N+1-th month and predicts N+2-th, etc. We are considering implementing this in the next versions.
  * For batch traiing, the train set is not evaluated, because, again, it takes too much time for the big datasets. So what we will do is adding additional `display` subset, which is part of training, but on which the metrics are computed.
* `metrics_pool` to compute metrics
* `artificiers` to inject arbitrary code into training

Also, there is a `settings` object that contains some self-explainatory fields.


```python
from sklearn.metrics import roc_auc_score


def build_splitter():
    return bt.CompositionSplitter(
        bt.FoldSplitter(test_size=0.2),
        bt.FoldSplitter(test_size=0.2, decorate=True, test_name='display')
    )


task = bt.BatchedTrainingTask(
    splitter = build_splitter(),
    batcher = batcher,
    model_handler=TorchHandler(),
    metric_pool = bt.MetricPool().add_sklearn(roc_auc_score),
    settings = bt.TrainingSettings(epoch_count=1)
)

result = task.run(ibundle_fixed)
```

    2022-06-27 19:29:58.428635+00:00 INFO: Training starts. Info: {}
    2022-06-27 19:29:58.430847+00:00 INFO: Ensuring/loading bundle. Bundle before:
    <tg.common.ml.batched_training.data_bundle.IndexedDataBundle object at 0x7f99f8e024f0>
    2022-06-27 19:29:58.433336+00:00 INFO: Bundle loaded
    {'index': {'shape': (891, 5), 'index_name': 'PassengerId', 'columns': ['Name', 'Ticket', 'Cabin', 'Survived', 'priority'], 'index': [1, 2, 3, 4, 5, '...']}, 'passengers': {'shape': (891, 4), 'index_name': 'Name', 'columns': ['Sex', 'Age', 'SibSp', 'Parch'], 'index': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath (Lily May Peel)', 'Allen, Mr. William Henry', '...']}, 'tickets': {'shape': (681, 3), 'index_name': 'Ticket', 'columns': ['Pclass', 'Fare', 'Embarked'], 'index': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450', '...']}}
    2022-06-27 19:29:58.438464+00:00 INFO: Index frame is set to index, shape is (891, 5)
    2022-06-27 19:29:58.445809+00:00 INFO: Skipping late initialization
    2022-06-27 19:29:58.447953+00:00 INFO: Preprocessing bundle by batcher
    2022-06-27 19:29:58.472113+00:00 INFO: Splits: train 712, test 179, display 143
    2022-06-27 19:29:58.474187+00:00 INFO: New training. Instantiating the system
    2022-06-27 19:29:58.478148+00:00 INFO: Fitting the transformers
    2022-06-27 19:29:58.702007+00:00 INFO: Instantiating model
    2022-06-27 19:29:58.752867+00:00 INFO: Initialization completed
    2022-06-27 19:29:58.759070+00:00 INFO: Epoch 0 of 1
    2022-06-27 19:29:58.759905+00:00 INFO: Training: 0/8
    2022-06-27 19:29:58.894238+00:00 INFO: Training: 1/8
    2022-06-27 19:29:58.968001+00:00 INFO: Training: 2/8
    2022-06-27 19:29:59.029186+00:00 INFO: Training: 3/8
    2022-06-27 19:29:59.145526+00:00 INFO: Training: 4/8
    2022-06-27 19:29:59.265845+00:00 INFO: Training: 5/8
    2022-06-27 19:29:59.431313+00:00 INFO: Training: 6/8
    2022-06-27 19:29:59.588380+00:00 INFO: Training: 7/8
    2022-06-27 19:29:59.764171+00:00 INFO: test: 0/2
    2022-06-27 19:29:59.935129+00:00 INFO: test: 1/2
    2022-06-27 19:30:00.151570+00:00 INFO: display: 0/2
    2022-06-27 19:30:00.344605+00:00 INFO: display: 1/2
    2022-06-27 19:30:00.527466+00:00 INFO: ###roc_auc_score_test:0.7768115942028986
    2022-06-27 19:30:00.531099+00:00 INFO: ###roc_auc_score_display:0.6655913978494623
    2022-06-27 19:30:00.537008+00:00 INFO: ###loss:0.2480677142739296
    2022-06-27 19:30:00.547391+00:00 INFO: ###iteration:0


As you can see, `TrainingTasks` logs quite extensively on the initialization process, so in case of error it's relativaly easy to understand the source of error. 

Now, let's disable `Logger` and run training for the longer time:


```python
from tg.common import Logger
Logger.disable()

task = bt.BatchedTrainingTask(
    splitter = build_splitter(),
    batcher = batcher,
    model_handler=TorchHandler(),
    metric_pool = bt.MetricPool().add_sklearn(roc_auc_score),
    settings = bt.TrainingSettings(epoch_count=10)
)

result = task.run(ibundle_fixed)

pd.DataFrame(result['output']['history']).set_index('iteration').plot()
pass
```


    
![png](README_images/tg.common.ml.batched_training_output_59_0.png?raw=true)
    


Feel free to explore other fields of `result` as well. We hope the name of the fields are quite self-explanatory.

Training can be continued, in this case history persists and the models are not recreated.


```python
task.settings.continue_training=True
result = task.run(ibundle_fixed)
pd.DataFrame(result['output']['history']).set_index('iteration').plot()
pass
```


    
![png](README_images/tg.common.ml.batched_training_output_62_0.png?raw=true)
    


You don't have to use the same bundle for continuation. So it's totally feasible to tune our models with newly available data instead of retrain them from scratch every time. 


### Late initialization

Late initialization allows you to:
* alter data in the bundle when they were loaded
* initialize or modify arbitrary fields of the task, based on loaded data

The main use case is that the model might require some tweaking depending on the input data, or the input data might require tweaking itself. For instance, in our case, it's a row priority, which we hacked into the bundle while discussing `BatchingStrategy`. 

The proper way is to compute this field in late initialization. This method accepts task, bundle and environment (for logging purposes), and can modify any of those.

Actually, late initialization is also implemented in single-frame training, but for this one we don't really have the use cases yet.


```python
def late_initialization(task, ibundle):
    ibundle.index_frame['priority'] = bt.PriorityRandomBatcherStrategy.make_priorities_for_even_representation(
        ibundle.index_frame,
        'Survived'
    )


task = bt.BatchedTrainingTask(
    splitter = build_splitter(),
    batcher = batcher,
    model_handler=TorchHandler(),
    metric_pool = bt.MetricPool().add_sklearn(roc_auc_score),
    settings = bt.TrainingSettings(epoch_count=10),
    late_initialization=late_initialization
)

result = task.run(ibundle)
pd.DataFrame(result['output']['history']).set_index('iteration').plot()
pass
```


    
![png](README_images/tg.common.ml.batched_training_output_65_0.png?raw=true)
    


### Prediction

In many cases, the model can be directly applied to the data to make predictions. To do this, simply construct the DataBundle with the data for prediction and apply method predict to it.


```python
prediction = task.predict(ibundle.change_index(ibundle.index_frame.iloc[:5]))
prediction
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted</th>
      <th>true</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.394233</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.671026</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.483621</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.651865</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.387574</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



In some cases, you don't really need the model to predict anything, because, for instance, you're actually interested in scores that are produced as the intermediate values in your network. In this case, you will need to write the prediction procedure yourself. Keep in mind that:

* Batcher is as essensial for the prediction as is model. Batcher contains all the extractors that were fitted during the training, and the model simply won't work without them.
* There is some additional initialization of the bundle that may be needed, so observe the `predict` method for exact sequence of the initialization steps.

## Troubleshooting

Batched training is a lot more complicated process than single-frame training. Generally, there are two major sources of errors:
* Something is wrong with batchers/extractors
* Something is wrong with the network process: dimensions/shapes do not match, data types are unexpected, etc.

### Separating batching from model

Errors in batcher can be debugged by separately running extractors and evaluating their results, as we did in this demo. Errors in network, however, are harder to track, as the network is fed with the batcher's output. To debug the network, we offer the access to initialization and batch creation separately from training.

This method will allow you to initialize everything and get the batch:


```python
task = bt.BatchedTrainingTask(
    splitter = build_splitter(),
    batcher = batcher,
    model_handler=TorchHandler(),
    metric_pool = bt.MetricPool().add_sklearn(roc_auc_score),
    settings = bt.TrainingSettings(epoch_count=10)
)
batch = task.generate_sample_batch(ibundle_fixed)
```

This batch then can be used on different levels to debug network and handler:


```python
task.model_handler.network(batch['features'])[:5]
```




    tensor([[0.4233],
            [0.4036],
            [0.4089],
            [0.4441],
            [0.4253]], grad_fn=<SliceBackward>)




```python
task.model_handler.predict(batch).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted</th>
      <th>true</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>423</th>
      <td>0.423282</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0.403562</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>431</th>
      <td>0.408860</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>788</th>
      <td>0.444078</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>835</th>
      <td>0.425296</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Debug mode

`BatchingTrainingTask` has a `debug` argument, which forces the task to keep the intermediate data as a field of the class. **Never** do it in production, as the intermediate data also contain the bundle, so pickling the task (which is an artefact of the training) will be impossible with any real data. 

However, with toy datasets such as we have, it's very useful to look at the intermediate dataframes when debugging errors. For instance, let's check if the `test` and `display` splits are related to the train split the way we expect:


```python
task = bt.BatchedTrainingTask(
    splitter = build_splitter(),
    batcher = batcher,
    model_handler=TorchHandler(),
    metric_pool = bt.MetricPool().add_sklearn(roc_auc_score),
    settings = bt.TrainingSettings(epoch_count=1),
    late_initialization=late_initialization,
    debug = True
)
task.run(ibundle)
pass
```

We can, for instance, observe the modification of the bundle, performed in the `late_initialization`:


```python
task.data_.original_ibundle.index_frame.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Survived</th>
      <th>priority</th>
    </tr>
    <tr>
      <th>PassengerId</th>
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
      <td>Braund, Mr. Owen Harris</td>
      <td>A/5 21171</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.001821</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>PC 17599</td>
      <td>C85</td>
      <td>1.0</td>
      <td>0.002924</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Heikkinen, Miss. Laina</td>
      <td>STON/O2. 3101282</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.002924</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>113803</td>
      <td>C123</td>
      <td>1.0</td>
      <td>0.002924</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Allen, Mr. William Henry</td>
      <td>373450</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.001821</td>
    </tr>
  </tbody>
</table>
</div>



Or, we may ensure that the splits are consistant with our expectations:


```python
(
    task.data_.result.test_splits['test'].isin(task.data_.result.train_split).mean(),
    task.data_.result.test_splits['display'].isin(task.data_.result.train_split).mean()
)

```




    (0.0, 1.0)



Or, if we're interested in the current batch:


```python
task.data_.batch['index'].groupby('Survived').size()
```




    Survived
    0.0    57
    1.0    37
    dtype: int64



In general, if the training fails at some step, there is a good chance that the input for this step is stored in `data_` field, so you may debug the failing step separately.

## Advanced techniques 

### Minibatches 

As mentioned earlier, sometimes it makes sense to subdivide batch into smaller mini-batches, to optimize both the efficiency of the training and the quality of the result. This can be achieved by using `BatchedTrainingTask.settings.batch_size` and `mini_batch_size` fields.

Typically, we opt for the following strategy:
  * Finding the biggest `batch_size` that fits the network
  * Setting something like 200 to `mini_batch_size`, maybe additionally checking values 20 and 2000. In our experience, the network is not sensitive to this parameter as long as it stays within reasonable range.
  
### Multi-task bundles

Sometimes we may pose different tasks with essentially same data, e.g., different prediction tasks based on the same NLP corpus. In this case, you make generate several index frames with different names, and set `BatchedTrainingTask.index_name_in_bundle` when running the task.


### Heavy precomputing in extractors

Sometimes `Extractor` needs to perform a heavy computation. For instance, in NLP task we may want to compute for each word it's frequency in the text. Or, in customer-to-article relation, we may want to compute an average performance of the article in the previous months. Doing so for each batch would be impractical, it would be much better to do once.

However, doing so in `fit` is wrong for two reasons:
* Storing data in the model will increase its size and generally break the border between data and model, which we try to maintain very hard in TG.
* When predicting, `fit` obviously should not be called, but such intermediate data should be computed.

Therefore, `preprocess_bundle` is available in the `Extractor`, where such computation may be performed.


### Hyperparameter tuning

Implemented exactly the same way as in single-frame training. 

In practice, we never used this functionality, nor did we debug it with hyperopt. We opted for manual tuning in the following way:
  * there is a `build_task` method that accepts essential parameters
  * we run nested loop over parameters, create task for every combination and run the task in the cloud





# 3.3.1. Batched training with torch (tg.common.ml.batched_training.torch)

## Overview

`tg.common.ml.batched_training` is very generic and flexible, allowing to orchestrate the training process by using settings and component classes. It greatly reduces the amount of code that needs to be written to train one model, and does not limit the technology stack for the models themselves. 

Unfortunately, there is still _a lot_ of code that needs to be written to instantiate the training task, namely, the `ModelHandler`: the way we have written it in the demo, any small change in network architecture requires the code update. We strongly prefer parameter-based model's definition over code based, as:

  * It's easier to compare and reproduce the results of the experiments
  * It's testable and reusable

Therefore, we wanted to enable a parameter-based definition at least for some models, choosing `PyTorch`:
  * `optimizer` and `loss` in this case do not require coding, as the type names may be used
  * networks creation does not necesserily require coding and can be replaced by _factories_.

Also, we have noticed that some elements of the functionality is not really required in practice:
  * instead of defining splitters, it's more practical to put the values directly in the `index_frame`, thus ensuring all the networks versions are going to be compared over the same sets
  * `BatchingStrategy` can simply be defined by the presence of `priority` column in the `index_frame` (either defined in bundle preparation or in `late_initialization`)
  
  
In essense, what remains really important for the training of `torch` networks, is a pair of two factories:
  * the first one produce the extractors, given the bundle; it also performs necessary bundle tuning. It does requires coding, if the extractors are dependent on the bundle, which is usually not the case, or if tuning is needed.
  * the second one produce the networks, given the batch. It requires coding, if the network is build from predefined blocks.
  
This functionality is implemented in a slim wrapper around `BatchedTrainingTask`, namely, `TorchTrainingTask`. This demonstration will show how to use it. First, let us load the bundle and define the `train`, `display` and `test` splits.


```python
from tg.common.ml import batched_training as bt
from sklearn.model_selection import train_test_split
import numpy as np

bundle = bt.DataBundle.load('temp/bundle')

train, test = train_test_split(bundle.index.index, stratify=bundle.index['Survived'], test_size=0.2)
train, display = train_test_split(train, stratify = bundle.index.loc[train]['Survived'], test_size=0.2)
bundle.index['split'] = np.where(
    bundle.index.index.isin(train),
    'train',
    np.where(
        bundle.index.index.isin(display),
        'display',
        'test'
    ))
bundle.index.groupby(['split','Survived']).size()
```




    split    Survived
    display  0.0          88
             1.0          55
    test     0.0         110
             1.0          69
    train    0.0         351
             1.0         218
    dtype: int64



Second, let's remember the extractors we have defined in the previous demo, and define the extractor factory:


```python
from tg.common.ml import dft
from tg.common.ml.batched_training import torch as btt

tfac = dft.DataFrameTransformerFactory.default_factory

extractors = [
    bt.PlainExtractor.build(name='label').apply(take_columns='Survived'),
    bt.PlainExtractor.build(name='cabin').index().apply(transformer=tfac(), take_columns='Cabin'),
    bt.PlainExtractor.build('passengers')
                       .index()
                       .join(frame_name='passengers', on_columns='Name')
                       .apply(transformer=tfac()),
    bt.PlainExtractor.build(name='tickets')
                    .index()
                    .join(frame_name='tickets', on_columns='Ticket')
                    .apply(transformer=tfac())
]

extractor_factory = btt.PredefinedExtractorFactory(*extractors)
```

As a third step, we will define a network factory:


```python
network_factory = (btt.FullyConnectedNetwork.Factory(sizes = [10,1])
                   .prepend_extraction(input_frames = ['tickets', 'passengers', 'cabin']))

```

We will explain the architecture behind this definition a bit later. For now, let's define and run `TorchTrainingTask`:


```python
from sklearn.metrics import roc_auc_score
from tg.common import Logger
import pandas as pd

task = btt.TorchTrainingTask(
    bt.TrainingSettings(
        epoch_count=10,
        batch_size=50
    ),
    btt.TorchTrainingSettings(
        optimizer_ctor = btt.OptimizerConstructor('torch.optim:SGD',lr=0.5),
        loss_ctor = btt.ModelConstructor('torch.nn:MSELoss')
    ),
    extractor_factory,
    network_factory,
    bt.MetricPool().add_sklearn(roc_auc_score)
)

Logger.disable()


result = task.run(bundle)

pd.DataFrame(result['output']['history']).set_index('iteration').plot()
```




    <AxesSubplot:xlabel='iteration'>




    
![png](README_images/tg.common.ml.batched_training.torch_output_8_1.png?raw=true)
    


We can see that definition of `TorchTrainingTask` is indeed performed with only existing components and does not require much coding.

## Network factories

Network factories are currently more like proof of concept than a comprehensive set of all possible architectures. Still, we find the concept useful. The features are:

* Networks accept batches instead of tensors
* Networks adopt to the batches by choosing the input size so that it matches

This is achieved with the following architecture:

* We have "normal" networks, like `FeedForwardNetwork` or `LSTMNetwork`. These are normal `torch` modules that can be used "as is". 
* Each of such classes, however, has a nested `Factory` class, that is the factory creating the network from the input batch. The input batch is expected to be a torch tensor.
* There are also ways to organize networks (and factories) into structures, for instance, `FeedForwardNetwork` is a network that accepts several networks and pass the signal sequencially. Correspondindly, `FeedForwardNetwork.Factory` accepts several factories, creates instances for them, and in the end -- the instance for `FeedForwardNetwork`.
* For convienience, `FeedForwardNetwork.Factory` has `prepend_extraction` methodm that returns a `FeedForwardNetwork.Factory` with two nested factories: `ExtractingNetwork.Factory` and `FeedForwardNetwork.Factory`. Here `ExtractingNetwork.Factory` create a fake network, that accepts batch, translates the frames into `torch` tensors and concatenates them.



# 3.3.2. Batched training with contexts (tg.common.ml.batched_training.context)

This demo will cover training on data that are _contextual_. By that we mean cases, when some of the samples form relations, and to make a prediction about sample ***i***, the network must take into consideration all the samples, related to this sample ***i***.

The typical example is Natural Language Processing. Often, we cannot really do anything with an individual word in the sentence, we need to consider other words in its vicinity. Most often, ***N*** words to the left form this vicinity, but not only: if a parse tree of the sentence is available, we may consider "brothers" of the word, or parents, etc. 

Contextual data are also possible in sales. If we try to predict the customer-to-article relation, there are many assotiated contexts: previous purchases of the customer, other articles in the order, historical performance of the article, etc.

This demo will discuss ways of organizing the contexts in the various ways, including using of recurrent neural networks (LSTM)

## Setup

We will consider a toy task from NLP. Assume we have ***n*** words, and sentences of ***m*** such words. Some of these sentences belong to the "good" subset ***L***, and some don't. The task is to build a classifier for ***L***.

First, to establish the baseline, we will build the network without any contexts. For that, we will first build a training data as a dataframe with words in columns:


```python
import numpy as np
import pandas as pd
import os

def generate_task(word_length, alphabet):
    tuples = [ (c,) for c in alphabet]
    result = list(tuples)
    for i in range(word_length-1):
        result = [ t+r for t in tuples for r in result]
    df = pd.DataFrame(result, columns=[f'word_{i}' for i in range(word_length)])
    df[f'label'] = np.random.randint(0,2,df.shape[0])
    df.index.name = 'sentence_id'
    df['split'] = 'display'
    return df

df = pd.read_parquet('lstm_task.parquet')
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word_0</th>
      <th>word_1</th>
      <th>word_2</th>
      <th>word_3</th>
      <th>label</th>
      <th>split</th>
    </tr>
    <tr>
      <th>sentence_id</th>
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
      <th>0</th>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>display</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>1</td>
      <td>display</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>display</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>0</td>
      <td>display</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>0</td>
      <td>display</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (81, 6)



The following code will create a `BatchedTrainingTask` for this classification task, similarly to what we've seen in the previous demos. It will also output a batch for the task.


```python
from tg.common.ml import dft
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import torch as btt
from sklearn.metrics import roc_auc_score
from tg.common import Logger

Logger.disable()

def create_plain_task(df, epochs = 500):
    label_extractor = bt.PlainExtractor.build('label').index().apply(take_columns='label')
    features_extractor = bt.PlainExtractor.build('features').index().apply(
        take_columns=[f for f in df.columns if f.startswith('word')],
        transformer=dft.DataFrameTransformerFactory.default_factory()
    )
    extractor_factory = btt.PredefinedExtractorFactory(label_extractor, features_extractor)
    network_factory = btt.FullyConnectedNetwork.Factory(sizes=[100,1]).prepend_extraction('features')
    task = btt.TorchTrainingTask(
        bt.TrainingSettings(epoch_count = epochs),
        btt.TorchTrainingSettings(),
        extractor_factory,
        network_factory,
        bt.MetricPool().add_sklearn(roc_auc_score)
    )
    return task


plain_bundle = bt.DataBundle(index=df)
task = create_plain_task(df)
task.generate_sample_batch(plain_bundle)['features'].head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word_0_A</th>
      <th>word_0_B</th>
      <th>word_0_C</th>
      <th>word_1_A</th>
      <th>word_1_B</th>
      <th>word_1_C</th>
      <th>word_2_A</th>
      <th>word_2_B</th>
      <th>word_2_C</th>
      <th>word_3_A</th>
      <th>word_3_B</th>
      <th>word_3_C</th>
    </tr>
    <tr>
      <th>sentence_id</th>
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
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



We won't have any test data. As ***L*** is random, it's impossible to predict the status of the word without seeing it. So effectively, the neural network just needs to memorize the table. 
  

The following code will run the training task on the data multiple times and build a plot of `roc_auc` metric improvement over time for each of them:


```python
from yo_fluq_ds import *

def get_roc_auc_curve(name, task, bundle):
    result = task.run(bundle)
    series = pd.DataFrame(result['output']['history']).roc_auc_score_display
    series.name = str(name)
    return series

curves = (Query
          .en(range(5))
          .feed(fluq.with_progress_bar())
          .select(lambda z: get_roc_auc_curve(z, create_plain_task(df, epochs=500), bt.DataBundle(index=df)))
          .to_list()
         )
pd.DataFrame(curves).transpose().plot()
pass
```


      0%|          | 0/5 [00:00<?, ?it/s]



    
![png](README_images/tg.common.ml.batched_training.context_output_10_1.png?raw=true)
    


Note that:

* Training effectively stabilizes on ~200 iterations. We were unable to achieve significant improvement after this point.
* The quality is far from 100%. The amount of neurons in the network are compatible with the amount of samples, and, given that the samples are random, it's not surprising. 

It is probably possible to achieve higher metric on this task, but right now it's not our goal. 

The quality depends heavily on the training data: some of the random languages $L$ offer better performance than others. The following code was used to find a language with decent performance:



```python
def find_good_language(N):
    tasks = []
    rocs = []
    for i in Query.en(range(N)).feed(fluq.with_progress_bar()):
        df = generate_task(4, ['A','B','C'])
        tasks.append(df)
        roc = get_roc_auc_curve(i, create_plain_task(df), bt.DataBundle(df, {}))
        value = roc.iloc[-1]
        rocs.append(value)
        print(value, end=' ')
    
    s = pd.Series(rocs).sort_values()
    winner = s.index[-1]
    val = s.iloc[-1]
    tasks[winner].to_parquet(f'lstm_task_{val}.parquet')
    return s
```

## Contexts

The representation we used in the previous section is flawed due to several reasons:

* Sequential data usually have different length. Placing them in the columns of the dataframe causes sparsity. 
* Sequential data may contain multiple columns for each position, i.e. morphological features or word2vec for words. Placing them in the columns of the dataframe creates a hierarchy of the columns.
* Pandas is not really covenient to perform column-based operations, that will mean loops and other low-performative python logic.
* Samples may enter in more than one relations, and it's unpractical to build such table for each of them.

This is why it's far more convenient to use a sequential representation: each row is a sample (word in our case), but there is also a structural information, incorporated in the bundle, that represents relation of the samples. 

Let's translate our language ***L*** to this new format:


```python
def translate_to_sequential(df):
    words = [c for c in df.columns if c.startswith('word_')]
    context_length = len(words)

    cdf = df[words].unstack().to_frame('word').reset_index()
    cdf = cdf.rename(columns=dict(level_0= 'word_position'))
    cdf.word_position = cdf.word_position.str.replace('word_','').astype(int)
    cdf = cdf.sort_values(['sentence_id','word_position'])
    cdf['word_id'] = list(range(cdf.shape[0]))
    cdf = cdf[['word_id','sentence_id','word_position','word']]
    cdf.index = list(cdf['word_id'])
    
    idf = df[['label']].reset_index()
    idf['split'] = 'display'
    idf.index.name='sample_id'
    bundle = bt.DataBundle(index = idf, src=cdf)
    bundle.additional_information.context_length = context_length
    return bundle

db = translate_to_sequential(df)
```

`src` frame contains all word in the sentences;


```python
db.src.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word_id</th>
      <th>sentence_id</th>
      <th>word_position</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>



This structure is what we currently think the best approach for NLP:

* `word_id` is a unique, always-increasing `id` of the occurence of the word in the text
* `sentence_id` is a unique, always-increasing `id` of the word/sentence.
* `word_position` additionally positions words within sentences
* Additional indexations (like `paragraph_id`, `sentence_position`, etc) are possible.



```python
db.index.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentence_id</th>
      <th>label</th>
      <th>split</th>
    </tr>
    <tr>
      <th>sample_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>display</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>display</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>display</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>display</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>display</td>
    </tr>
  </tbody>
</table>
</div>



**Note**: indexation might appear excessive. We just follow the practice from other NLP tasks: there, not all the sentences in the corpus may appear in `index_frame`; as for `src`, in some cases it's handy to have `word_id` as a column, and in other cases -- as index, therefore, index of `src` simply duplicates the `word_id` column.

Now, our task is to build back the features for each sentence, by taking all the previous words in this sentence, transforming them in the features individually and then combining. This procedure is done by four entities, three basic are:
* `ContextBuilder` builds the context, i.e. relation from one instance of `index` entity to several other entities (not necessarily from `index`). 
* `Extactor` extracts features for each sample in the context.
* `Aggregator` then organizes the features so they are in the format, consumable by the network (2D dataframe)

There might be multiple extractors and aggregators for each context. Therefore, we add `Finalizer`, that concatenates aggregator's results, and also controls shape of the resulting dataframe, e.g. that even the samples with empty context receive their row in the features, and that all the columns are in their exact place.

Let's first cover the `ContextBuilder`:


```python
from tg.common.ml.batched_training import context as btc

class SentenceContextBuilder(btc.ContextBuilder):
    def build_context(self, ibundle, context_size):
        df = ibundle.index_frame[['sentence_id']]
        df = df.merge(ibundle.bundle.src.set_index('sentence_id'), left_on='sentence_id', right_index=True)
        df = df[['word_position','word_id','word']]
        df = df.loc[df.word_position<context_size]
        df = df.set_index('word_position', append=True)
        return df

ibundle = bt.IndexedDataBundle(db.index, db)
ibundle_sample = ibundle.change_index(ibundle.index_frame.iloc[:3])
context_builder = SentenceContextBuilder()
context_builder.fit(ibundle)
context_builder.build_context(ibundle_sample, 4)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>word_id</th>
      <th>word</th>
    </tr>
    <tr>
      <th>sample_id</th>
      <th>word_position</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">0</th>
      <th>0</th>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>A</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">1</th>
      <th>0</th>
      <td>4</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>B</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">2</th>
      <th>0</th>
      <td>8</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



As promised, it builds a relation from each `sample` to several rows in `src`, by different offsets. 

Now, we can use normal `Exctractor` to extract features. 

Also, we will use aggregator that simply concatenates the features for different samples in context, and `PandasFinalizer`, which should be always for every context extraction in 2D format. 


```python
def build_context_extractor(context_length, aggregator):
    context_extractor = btc.ContextExtractor(
        name = 'features',
        context_size = context_length,
        context_builder = SentenceContextBuilder(),
        feature_extractor_factory = btc.SimpleExtractorToAggregatorFactory(
            bt.PlainExtractor.build('word').index().apply(
                take_columns=['word'], 
                transformer = dft.DataFrameTransformerFactory.default_factory()
            ),
            aggregator
        ),
        finalizer = btc.PandasAggregationFinalizer(
            add_presence_columns=False
        ),
        debug = True
    )
    return context_extractor

context_extractor = build_context_extractor(
    db.additional_information.context_length,
    btc.PivotAggregator()
)
context_extractor.fit(ibundle)
context_extractor.extract(ibundle_sample)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f0a0_word_A_at_0</th>
      <th>f0a0_word_A_at_1</th>
      <th>f0a0_word_A_at_2</th>
      <th>f0a0_word_A_at_3</th>
      <th>f0a0_word_B_at_0</th>
      <th>f0a0_word_B_at_1</th>
      <th>f0a0_word_B_at_2</th>
      <th>f0a0_word_B_at_3</th>
      <th>f0a0_word_C_at_0</th>
      <th>f0a0_word_C_at_1</th>
      <th>f0a0_word_C_at_2</th>
      <th>f0a0_word_C_at_3</th>
    </tr>
    <tr>
      <th>sample_id</th>
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
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Note that it looks exactly as a batch in the very first section. But this time the dataframe is assembled by the components from `tg.common.ml.batched_training.context`, and it's shape and nature can be altered by changing these components, thus enabling different architectures for the networks (including the recurrent ones)

`ContextExtractor` performs non-trivial functionality, and stepwise debugging may be needed. For this, `debug` argument can be used, exactly as it was the case with `BatchedTrainingTask` (and it also should be **always off** in production due to the same reasons)

We can explore intermediate stages within `ContextExtractors`, e.g. the output of the `Extractor`:


```python
context_extractor.data_.feature_dfs['f0'].head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>word_A</th>
      <th>word_B</th>
      <th>word_C</th>
    </tr>
    <tr>
      <th>sample_id</th>
      <th>word_position</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">0</th>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Now, we will assemble the neural network to train on this data. 


```python
def create_sequential_task(context_extractor, network_factory, lr = 1, epochs = 100):
    task = btt.TorchTrainingTask(
        bt.TrainingSettings(epoch_count=epochs),
        btt.TorchTrainingSettings(
            optimizer_ctor=btt.OptimizerConstructor('torch.optim:SGD', lr=lr)
        ),
        btt.PredefinedExtractorFactory(
            context_extractor,
            bt.PlainExtractor.build('label').index().apply(take_columns='label')
        ),
        network_factory,
        bt.MetricPool().add_sklearn(roc_auc_score)
    )
    return task

task = create_sequential_task(
    build_context_extractor(
        db.additional_information.context_length,
        btc.PivotAggregator()
    ),
    btt.FullyConnectedNetwork.Factory([100,1]).prepend_extraction(['features'])
)
    
get_roc_auc_curve('', task, db).plot()
pass
```


    
![png](README_images/tg.common.ml.batched_training.context_output_30_0.png?raw=true)
    


So, the system achieved the same performance at the same time, as a naive implementation, confirming the correctness of the implementation (of course, the used classes are also covered by tests).

Let's now explore the task a bit further and see how the context length affects the performance:


```python
curves = []
for i in Query.en(range(1,db.additional_information.context_length+1)).feed(fluq.with_progress_bar()):
    task = create_sequential_task(
        build_context_extractor(i, btc.PivotAggregator()),
        btt.FullyConnectedNetwork.Factory([100,1]).prepend_extraction(['features'])
    )
    curves.append(get_roc_auc_curve(i, task, db))
pd.DataFrame(curves).transpose().plot()
pass
```


      0%|          | 0/4 [00:00<?, ?it/s]



    
![png](README_images/tg.common.ml.batched_training.context_output_32_1.png?raw=true)
    


Unsurprisingly, we see that if the context is smaller that the actual length of the sentences in our langauge $L$, the performance decreases.

`PivotAggregator` is the most memory-consuming way of representing the contextual data. In this example it's fine, but if context consists of dozens of samples, each having dozens of extracted columns, `PivotAggregator` will produce a very huge matrix that may overfill the memory. 

This is why you may also want to use other aggregators. For instance, `GroupByAggregator` will process the `features` dataframe with grouping by `sample_id` and applying the aggregating functions:


```python
context_extractor = build_context_extractor(i, btc.GroupByAggregator(['mean','max']))
task = create_sequential_task(
    context_extractor,
    btt.FullyConnectedNetwork.Factory([100,1]).prepend_extraction('features')
)
batch = task.generate_sample_batch(db)
Query.en(context_extractor.data_.agg_dfs.values()).first().head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word_A_mean</th>
      <th>word_A_max</th>
      <th>word_B_mean</th>
      <th>word_B_max</th>
      <th>word_C_mean</th>
      <th>word_C_max</th>
    </tr>
    <tr>
      <th>sample_id</th>
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
      <th>0</th>
      <td>1.00</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.75</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.75</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.75</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.50</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Let's evaluate performance of this system as well


```python
curves = []
for i in Query.en(range(1,db.additional_information.context_length+1)).feed(fluq.with_progress_bar()):
    task = create_sequential_task(
        build_context_extractor(i, btc.GroupByAggregator(['mean', 'max'])),
        btt.FullyConnectedNetwork.Factory([100,1]).prepend_extraction('features')
    )
    curves.append(get_roc_auc_curve(i, task, db))
pd.DataFrame(curves).transpose().plot()
pass
```


      0%|          | 0/4 [00:00<?, ?it/s]



    
![png](README_images/tg.common.ml.batched_training.context_output_37_1.png?raw=true)
    


Of course, it does not work well, because we effectively destroy the information about the order of the letter in the word. 

But maybe we can invent some custom aggregator, which averages elements of the context with different weights, so the information is somehow preserved:


```python
class CustomAggregator(btc.ContextAggregator):
    def aggregate_context(self, features_df):
        names = features_df.index.names
        if names[0] is None:
            raise ValueError('There is `None` in the features df index. This aggregator requires you to set the name for index of your samples')
        columns = features_df.columns
        df = features_df.reset_index()
        for c in columns:
            df[c] = df[c]/(df[names[1]]+1)
        df = df.groupby(names[0])[columns].mean()
        return df
    
context_extractor = build_context_extractor(i, CustomAggregator())
task = create_sequential_task(
    context_extractor,
    btt.FullyConnectedNetwork.Factory([100,1]).prepend_extraction('features')
)
batch = task.generate_sample_batch(db)
Query.en(context_extractor.data_.agg_dfs.values()).first().head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word_A</th>
      <th>word_B</th>
      <th>word_C</th>
    </tr>
    <tr>
      <th>sample_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.520833</td>
      <td>0.000000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.458333</td>
      <td>0.062500</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.458333</td>
      <td>0.000000</td>
      <td>0.0625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.437500</td>
      <td>0.083333</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.375000</td>
      <td>0.145833</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
curves = []
for i in Query.en(range(1,db.additional_information.context_length+1)).feed(fluq.with_progress_bar()):
    task = create_sequential_task(
        build_context_extractor(i, CustomAggregator()),
        btt.FullyConnectedNetwork.Factory([100,1]).prepend_extraction('features')
    )
    curves.append(get_roc_auc_curve(i, task, db))
pd.DataFrame(curves).transpose().plot()
pass
```


      0%|          | 0/4 [00:00<?, ?it/s]



    
![png](README_images/tg.common.ml.batched_training.context_output_40_1.png?raw=true)
    


Well, sounded good, didn't work.

## Caching contextual features

In some cases, contextual features are very large. If we had context length of 10 and 10 letters, we would have 100-fold increase of the size for index in the intermediate tables. This might overfill the memory, and this is why we compute the contextual features for the batch instead of the whole index.

However, that might bring the performance issues, as the contextual features are computed repeatedly, in each epoch. 

To offer a balance for that, we offer `PrecomputingExtractor`. This is a decorator that runs an arbitrary extractor in the beginning of the training, fits it, computes the features for the whole index and stores in the bundle.


```python
import copy 

context_extractor = build_context_extractor(i, btc.GroupByAggregator(['mean','max']))
precompuring_extractor = bt.PrecomputingExtractor(
    name = "precomputed_features",
    inner_extractor = context_extractor)
ibundle_test = copy.deepcopy(ibundle)
precompuring_extractor.fit(ibundle)
ibundle.bundle.precomputed_features.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f0a0_word_A_mean</th>
      <th>f0a0_word_A_max</th>
      <th>f0a0_word_B_mean</th>
      <th>f0a0_word_B_max</th>
      <th>f0a0_word_C_mean</th>
      <th>f0a0_word_C_max</th>
    </tr>
    <tr>
      <th>sample_id</th>
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
      <th>0</th>
      <td>1.00</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.75</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.75</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.75</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.50</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



As we see, the features were precomputed and added in `fit` method, which is called once before all the training start. The implementation will work for `BatchedTrainingTask.predict` method, as `PrecomputingExtractor`, when fitted, computes features in `preprocess_bundle` method, called by `predict`.

## LSTM

LSTM network, and, generally, recurrent neural networks are designed specifically for the tasks with context. So it's natural to try them for our tasks.

These networks, however, require 3-dimensional tensor as an input. So, we will need to modify the `ContextExtractor` in the following way:


```python
def build_lstm_extractor(context_length):
    context_extractor = btc.ContextExtractor(
        name = 'features',
        context_size = context_length,
        context_builder = SentenceContextBuilder(),
        feature_extractor_factory = btc.SimpleExtractorToAggregatorFactory(
            bt.PlainExtractor.build('word').index().apply(
                take_columns=['word'], 
                transformer = dft.DataFrameTransformerFactory.default_factory()
            )),
        finalizer = btt.LSTMFinalizer()
    )
    return context_extractor

lstm_extractor = build_lstm_extractor(ibundle.bundle.additional_information.context_length)
lstm_extractor.fit(ibundle)
value = lstm_extractor.extract(ibundle_sample)
```

Here, we 

* don't use the aggregator. Aggregators are converting the context with features into 2-dimensional structure, which is exactly what we want to avoid.
* use `LSTMFinalizer` instead of `PandasFinalizer`

`LSTMFinalizer` looks at the feature table and converts it into tensor with the right dimensions:


```python
value.tensor
```




    tensor([[[1., 0., 0.],
             [1., 0., 0.],
             [1., 0., 0.]],
    
            [[1., 0., 0.],
             [1., 0., 0.],
             [1., 0., 0.]],
    
            [[1., 0., 0.],
             [1., 0., 0.],
             [1., 0., 0.]],
    
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]])




```python
value.dim_names
```




    ['word_position', 'sample_id', 'features']




```python
value.dim_indices
```




    [[0, 1, 2, 3], [0, 1, 2], ['word_A', 'word_B', 'word_C']]



Now, we can train a neural network with `LSTMNetwork` component, which is a slim decorator over `LSTM` that manages it's output.


```python
def build_lstm_network_factory(lstm_size):
    return btt.FeedForwardNetwork.Factory(
        btt.ExtractingNetwork.Factory('features'),
        btt.LSTMNetwork.Factory(lstm_size),
        btt.FullyConnectedNetwork.Factory([1])
    )


task = create_sequential_task(
    build_lstm_extractor(db.additional_information.context_length),
    build_lstm_network_factory(10),
    epochs = 100,
)

get_roc_auc_curve('', task, db).plot()
pass
```


    
![png](README_images/tg.common.ml.batched_training.context_output_54_0.png?raw=true)
    


We see that after 100 iterations the quality is low, but growing, while previous systems have stabilized at this point. So, we will train this network for a longer time.


```python
task = create_sequential_task(
    build_lstm_extractor(db.additional_information.context_length),
    build_lstm_network_factory(10),
    epochs = 1000
)

get_roc_auc_curve('', task, db).plot()
pass
```


    
![png](README_images/tg.common.ml.batched_training.context_output_56_0.png?raw=true)
    


In all the cases we run this cell, the quality of the "plain" networks was surpassed, often reaching the 0.9 value without stabilization. It might be the sign that this network architecture is indeed superior in comparison with the other, but this is anyway not the point we're trying to make. The point is that the classes, used for the data transformation, are working correctly and are ready to be used in conjustion with LSTM and other recurrent neural networks.



# 4.1. Packages and Containers (tg.common.delivery.packaging)

## Overview

In this part, we will deliver the featurization job to a remote server and execute it there. This actually can be done with just few lines of code. But we will show a lot of the process "under the hood" to make you familiar with it, and to explain why do we have this setup.

Delivery is the most fundamental purpose of training grounds. It is extremely easy to write _some_ data science code, that is executable on your local machine. It is not so easy though to then deliver this code to a remote server (be it server for training or a web-server that exposes model to the world) so that everything continues to work.

Delivery in Training Grounds is built upon the following principles.

### Deliverables are pickled objects

We do not deliver chunks of code or notebooks. Instead, we deliver the objects that incapsulate this code.

The most simple way of doing it is write a class that contains all the required functionality in `run` method and deliver it. In the previous presentations you saw that the `FeaturizationJob` class is more complicated. We didn't have the functionality written in the run method; instead, this functionality was defined as a composition of smaller objects, according to SOLID principles. This is *not* a requirement of delivery subsystem, the delivery will work perfectly fine without any SOLID. 

When prototyping, we would recommend to stick to the simplest way, which is implementing everything in the `run` method. When the solution is developed enough, you may need to consider it's decomposition to the subclasses in order to provide testability and reusability.

### The source code is delivered alongside the objects

In many frameworks there is a backstage idea that the framework has a comprehensive set of bug-free basic objects, and any imaginable functionality we need can be composed from these. So the users would never need to write Python code ever again, instead they would write declarative descriptions of the functionality they need. In this mindset, the delivery of the source code can be performed with `pip install`.

This approach is not the one TG follows due to the various reasons:
* Frameworks seldom actually get to this stage of development
* Versioning is painful
* This mindset creates a complexity gap: to do something new, with no basic objects available, is a lot harder than using the constructor. In this regard, it is extremely important for us that the user can implement this prototyping functionality in the `run` method without using any complex architecture.

Therefore, the source code is changing rapidly. Publishing it via PiPy or `git` would create a very complicated setup, when delivery requires a lot of intermediate stages, such as commiting, pushing, tagging or publishing. 

The simpler solution is to package the current source code into a Python package, placing the pickled objects as resource inside this package. No external actions are required in this case: the object will be unseparable from the source code, thus preventing versioning issues.

### Multiple versions

We wanted different versions of a model to be able to run at the same time. But how can we do that, if the models are represented as packages? In Python, we cannot have two modules with the same name installed at the same time. Thus, they have to have different name. This is why Training Grounds itself is not a Python package, but a folder inside your project. 

Consider the file structure, recommended by TG:
```
/myproject/tg/
/myproject/tg/common/
/myproject/tg/mylibrary/
/myproject/some_other_code_of_the_project
```

When building a package, these files will be transfomed into something like:
```
/package_name/UID/
/package_name/UID/tg/
/package_name/UID/tg/common/
/package_name/UID/tg/mylibrary/
```

Note that everything outside of original `/myproject/tg/` folder is ignored. So outside of `tg` folder you can have data caches, temporal files, sensitive information (as long as it's not pushed in the repository) and so on. It will never be delivered anywhere. The corollary is that all the classes and functions you use in your object must be defined inside `/tg/` folder. Otherwise, they will not be delivered.

The name of the TG is actually `UID.tg`, with different UID in each package. Hence, several versions of TG can be used at the same time! But that brings another limitation that must be observed inside `tg` folder: all the references inside TG must be relative. They cannot refer to `tg`, because `tg` will become `{UID}.tg` in the runtime on the remote server.


### Hot Module Replacement

Now, the question arises, how to use this package. We cannot write something like:

```
from UID.tg import *
```

because the name `UID` is formed dynamically. 

The solution is to install the module during runtime. During this process, the name becomes known, and then we can dynamically import from the module. Of course, importing classes or methods would not be handy, but remember that deliverables are objects, and these objects are pickled as the module resources. So all we need to do is to unpickle these objects, and all the classes and methods will be loaded dynamically by unpickler. 

This work is done by `EntryPoint` class.

#### Note for advanced users

When package is created, we pickle the objects under the local version of TG, thus, the classes are unavoidably pickled as `tg.SomeClass`, but we want to unpickle them as `UID.tg.SomeClass`. How is this achived? Fortunately, pickling allows you to do some manipulations while pickling/unpickling, and so we just replace all `tg.` prefixes to `UID.tg.` while building a package (UID is already known at this time).

It is also possible to do same trick when unpickling: if you want to transfer the previously packaged object into the current `tg` version, this is possible. Of course, it's on your responsibility to ensure that current TG is compatible with an older version. Later we will discuss a use case for that.

## Packaging

Consider the following job we want to deliver to the remote server and execute there.


```python
from tg.common.datasets.featurization import FeaturizationJob, DataframeFeaturizer
from tg.common.datasets.selectors import Selector
from tg.common.datasets.access import MockDfDataSource
from tg.common import MemoryFileSyncer
import pandas as pd

mem = MemoryFileSyncer()

job = FeaturizationJob(
    name = 'job',
    version = 'v1',
    source = MockDfDataSource(pd.read_csv('titanic.csv')),
    featurizers = {
        'passengers': DataframeFeaturizer(row_selector = Selector.identity)
    },
    syncer = mem,
    location = './temp/featurization_job'
)

job.run()
mem.get_parquet(list(mem.cache)[0]).head()
```

    2022-06-27 19:39:59.245526+00:00 INFO: Featurization Job job at version v1 has started
    2022-06-27 19:39:59.247769+00:00 INFO: Fetching data
    2022-06-27 19:39:59.338822+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:39:59.409466+00:00 INFO: Uploading data
    2022-06-27 19:39:59.410950+00:00 INFO: Featurization job completed





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



For details about mentioned classes, we refer you to the previous demos. Essentialy, the code above just passes the `titanik.csv` file through the TG machinery, decomposes and reconstructs it again, without changing anything.

Let's build the package with this job:


```python
from tg.common.delivery.packaging import PackagingTask, make_package
import copy

packaging_task = PackagingTask(
    name = 'titanic_featurization',
    version = '1',
    payload = dict(job = job),
    silent = True
)

info = make_package(packaging_task)
desc = copy.deepcopy(info.__dict__)
del desc['properties']['dependencies'] # just for readbility
desc
```

    warning: no files found matching '*.yml' under directory 'titanic_featurization__1'
    warning: no files found matching '*.rst' under directory 'titanic_featurization__1'
    warning: sdist: standard file not found: should have one of README, README.rst, README.txt, README.md
    





    {'task': <tg.common.delivery.packaging.packaging_dto.PackagingTask at 0x7efc84636c70>,
     'module_name': 'titanic_featurization__1',
     'path': PosixPath('/home/yura/Desktop/repos/tg/temp/release/package/titanic_featurization__1-1.tar.gz'),
     'properties': {'module_name': 'titanic_featurization',
      'version': '1',
      'full_module_name': 'titanic_featurization__1',
      'tg_name': 'tg',
      'full_tg_name': 'titanic_featurization__1.tg'}}



Here `PackagingTask` defines all the properties of the package, and `make_package` creates the package. Normally, we don't use `silent=True` to see the intermediate steps.

**Note**: `name` and `version` here are the name and version in the sense of Python. 

If you create and install another package with the name `titanic_featurization` and higher version, the version 1 will be removed from the system - because Python does not allow you to have different versions of the same library at the same time. This is the way to go if you actually want to update the model.

If you want several models to be used at the same time, you need to incorporate the version inside name, e.g. `name=titanic_featurization_1`

Let us now install the created package. `make_package` stored a file in the local system, and now we will install it. In the code, it results in the `EntryPoint` object.


```python
from tg.common.delivery.packaging import install_package_and_get_loader

entry_point = install_package_and_get_loader(info.path, silent = True)
entry_point.__dict__
```

    Found existing installation: titanic-featurization 1
    Uninstalling titanic-featurization-1:
      Successfully uninstalled titanic-featurization-1





    {'module_name': 'titanic_featurization',
     'module_version': '1',
     'tg_module_name': 'titanic_featurization__1.tg',
     'python_module_name': 'titanic_featurization__1',
     'original_tg_module_name': 'tg',
     'resources_location': '/home/yura/anaconda3/envs/tg/lib/python3.8/site-packages/titanic_featurization__1/resources'}



Now we will load the job from the package. Note that the classes are indeed located in different modules.


```python
loaded_job = entry_point.load_resource('job')
print(type(job))
print(type(loaded_job))
```

    <class 'tg.common.datasets.featurization.simple.featurization_job.FeaturizationJob'>
    <class 'titanic_featurization__1.tg.common.datasets.featurization.simple.featurization_job.FeaturizationJob'>


## Containering

Although we could just run the package at the remote server via ssh, the more suitable way is to use Docker. Training Grounds defines methods to build the docker container out of the package.

`make_container` produces a very extensive output, so we will remove it with `clear_output` function.


```python
from tg.common.delivery.packaging import ContaineringTask, make_container
from IPython.display import clear_output

ENTRY_FILE_TEMPLATE = '''
import {module}.{tg_name}.common.delivery.jobs.ssh_docker_job_execution as feat
from {module} import Entry
import logging

logger = logging.getLogger()

logger.info("Hello, docker!")
job = Entry.load_resource('job')
job.run()
logger.info(job.syncer.get_parquet(list(job.syncer.cache)[0]).head())

'''

DOCKERFILE_TEMPLATE  = '''FROM python:3.7

{install_libraries}

COPY . /featurization

WORKDIR /featurization

COPY {package_filename} package.tar.gz

RUN pip install package.tar.gz

CMD ["python3","/featurization/run.py"]
'''

task = ContaineringTask(
    packaging_task = packaging_task,
    entry_file_name = 'run.py',
    entry_file_template=ENTRY_FILE_TEMPLATE,
    dockerfile_template=DOCKERFILE_TEMPLATE,
    image_name='titanic-featurization',
    image_tag='test'
)

make_container(task)
clear_output()
```

Now, we can run this container locally:


```python
!docker run titanic-featurization:test
```

    2022-06-27 19:40:27.972884+00:00 INFO: Featurization Job job at version v1 has started
    2022-06-27 19:40:27.973239+00:00 INFO: Fetching data
    2022-06-27 19:40:28.069563+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:40:28.130125+00:00 INFO: Uploading data
    2022-06-27 19:40:28.130755+00:00 INFO: Featurization job completed


This `make_container` function is not "standard" or "universal": it just allows building the containers that are suitable for Sagemaker tasks and featurization jobs. So if you need some more sophisticated containering, please check the source code of this function to understand how to create an analog for it. Most of the complicated job is done by packaging, so `make_container` really just fills templates with values and executes some shell commands.



# 4.2. Deliverable Jobs (tg.common.delivery.jobs)

## SSH/Docker routine

Fortunately, you don't really need to do packaging or containering yourself, because we have a higher level level interfaces to do that, which is `Routine` classes. 

Let's create a job that we are going to package:


```python
from tg.common.datasets.featurization import FeaturizationJob, DataframeFeaturizer
from tg.common.datasets.selectors import Selector
from tg.common.datasets.access import MockDfDataSource
from tg.common import MemoryFileSyncer
import pandas as pd

mem = MemoryFileSyncer()

job = FeaturizationJob(
    name = 'job',
    version = 'v1',
    source = MockDfDataSource(pd.read_csv('titanic.csv')),
    featurizers = {
        'passengers': DataframeFeaturizer(row_selector = Selector.identity)
    },
    syncer = mem,
    location = './temp/featurization_job'
)


```

Then, `SSHDockerJobRoutine` allows you to execute your jobs in the docker at the remote server to which you have ssh access.


```python
from tg.common.delivery.jobs import SSHDockerJobRoutine, DockerOptions
from tg.common.delivery.packaging import FakeContainerHandler

routine = SSHDockerJobRoutine(
    job = job,
    remote_host_address=None,
    remote_host_user=None,
    handler_factory = FakeContainerHandler.Factory(),
    options = DockerOptions(propagate_environmental_variables=[])
)
```

Most of the fields are specified to None, because we are not going to actually start the remote job with this notebook. `remote_host_address` and `remote_host_user` arguments are self-explainatory. 

As for `handler_factory`, this argument must be set to one of the factories that generate `ContainerHandler`. This `ContainerHandler` class must define a remote image name and tag, and perform push operations. `ContainerHandlers` are not included to the Training Grounds core, as they usually have some company-specific code.

`SSHDockerJobRoutine` has methods of running your code for debugging.

Using the `.attached` accesor, we can run job in the same Python process that your current code is executed. This is, of course, the fastest way to do that, and therefore it's preferrable to use this to debug for typos, wrong logic, etc.


```python
routine.attached.execute()
```

    2022-06-27 19:40:31.016162+00:00 INFO: Featurization Job job at version v1 has started
    2022-06-27 19:40:31.017706+00:00 INFO: Fetching data
    2022-06-27 19:40:31.075090+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:40:31.098569+00:00 INFO: Uploading data
    2022-06-27 19:40:31.099733+00:00 INFO: Featurization job completed


The `.local` accessor builds package and container, then executes the container locally. This step allows debugging the following things:

* If your job is serializable. This is usually achievable by not using `lambda` syntax.
* If all the code the job uses is located inside the TG folder, and if all the references are relative. If something is wrong, you will see the import error.
* If the environmental variables are carried to docker correctly. 
* If you have sufficient permissions to start docker
* etc.


This step allows you to check the deliverability of your work. As the output is quite big, we will remove it for better readability.


```python
from IPython.display import clear_output

routine.local.execute()
clear_output()
```

You can retrieve logs from the container with the following useful method. Note that logs printed via `logging` are placed in stderr instead of strdout.


```python
output, errors = routine.local.get_logs()
print(output)
```

    2022-06-27 19:40:54.883388+00:00 INFO: Welcome to Training Grounds. This is Job execution via Docker/SSH
    2022-06-27 19:40:54.930855+00:00 INFO: Executing job job version v1
    2022-06-27 19:40:54.931020+00:00 INFO: Featurization Job job at version v1 has started
    2022-06-27 19:40:54.931552+00:00 INFO: Fetching data
    2022-06-27 19:40:55.228919+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:40:55.310273+00:00 INFO: Uploading data
    2022-06-27 19:40:55.312840+00:00 INFO: Featurization job completed
    2022-06-27 19:40:55.313731+00:00 INFO: Job completed
    


`routine.remote` has the same interface as `routine.local`, and will run the container at the remote machine. The only problems you should have at these stage are permissions:
* to push to your docker registry
* to connect to the remote machine via SSH
* to execute `docker run` at the remote machine

## Summary 

In this demo, we delivered the job to the remote server and executed it there. That concludes the featurization-related part of the Training Grounds.

Note that the packaging and containering techniques are not specific for the featurization, and can process any code. In the subsequent demos, the same techniques will be applied to run the training on the remote server as well.



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
from sklearn.metrics import accuracy_score

task = sft.SingleFrameTrainingTask(
    data_loader = sft.DataFrameLoader('Survived'),
    model_provider=sft.ModelProvider(sft.ModelConstructor(
            'sklearn.linear_model:LogisticRegression',
            max_iter = 1000),
        transformer = dft.DataFrameTransformerFactory.default_factory(),
        keep_column_names=False),
    evaluator=sft.Evaluation.multiclass_classification,
    splitter=sft.FoldSplitter(),
    metrics_pool = sft.MetricPool().add_sklearn(accuracy_score)        
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

    2022-06-27 19:41:09.350865+00:00 INFO: Starting stage 1/1
    2022-06-27 19:41:09.558283+00:00 INFO: Saved artifact /home/yura/Desktop/repos/tg/temp/training_results/_20220627_214109_9995b5204d7d490781979ad50ceb2464/runs/0/result_df
    2022-06-27 19:41:09.559445+00:00 INFO: Saved artifact /home/yura/Desktop/repos/tg/temp/training_results/_20220627_214109_9995b5204d7d490781979ad50ceb2464/runs/0/metrics
    2022-06-27 19:41:09.562344+00:00 INFO: Saved artifact /home/yura/Desktop/repos/tg/temp/training_results/_20220627_214109_9995b5204d7d490781979ad50ceb2464/runs/0/info
    2022-06-27 19:41:09.572337+00:00 INFO: Saved artifact /home/yura/Desktop/repos/tg/temp/training_results/_20220627_214109_9995b5204d7d490781979ad50ceb2464/runs/0/model
    2022-06-27 19:41:09.576507+00:00 INFO: Saved artifact /home/yura/Desktop/repos/tg/temp/training_results/_20220627_214109_9995b5204d7d490781979ad50ceb2464/runs/0/training_task
    2022-06-27 19:41:09.580643+00:00 INFO: Saved artifact /home/yura/Desktop/repos/tg/temp/training_results/_20220627_214109_9995b5204d7d490781979ad50ceb2464/runs/0/train_split
    2022-06-27 19:41:09.589805+00:00 INFO: Saved artifact /home/yura/Desktop/repos/tg/temp/training_results/_20220627_214109_9995b5204d7d490781979ad50ceb2464/runs/0/test_splits
    2022-06-27 19:41:09.592056+00:00 INFO: Completed stage 1/1
    2022-06-27 19:41:09.600195+00:00 INFO: ###METRIC###accuracy_score_test:0.7910447761194029###
    2022-06-27 19:41:09.606074+00:00 INFO: ###METRIC###accuracy_score_train:0.8057784911717496###


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


We can view the resulting dataframe, and compute a confusion matrix, for instance:


```python
df = pd.read_parquet(attached_folder/'runs/0/result_df.parquet')
(df
 .loc[df.stage=='test']
 .groupby(['predicted','true'])
 .size()
 .to_frame('cnt')
 .pivot_table(index='predicted',columns='true',values='cnt').fillna(0)
)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>true</th>
      <th>0.0</th>
      <th>1.0</th>
    </tr>
    <tr>
      <th>predicted</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>140</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>28</td>
      <td>72</td>
    </tr>
  </tbody>
</table>
</div>



We can also unpickle model or the whole training task:


```python
FileIO.read_pickle(attached_folder/'runs/0/training_task.pkl')
```




    <tg.common.ml.single_frame_training.training_task.SingleFrameTrainingTask at 0x7fe68a5fb2e0>



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




    '_20220627_214109_80c8c40453a544cc99e69cd531aaeb71'




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
(df
 .loc[df.stage=='test']
 .groupby(['predicted','true'])
 .size()
 .to_frame('cnt')
 .pivot_table(index='predicted',columns='true',values='cnt').fillna(0)
)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>true</th>
      <th>0.0</th>
      <th>1.0</th>
    </tr>
    <tr>
      <th>predicted</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>140</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>28</td>
      <td>72</td>
    </tr>
  </tbody>
</table>
</div>



However, we cannot just open the `training_task`:


```python
import traceback
try:
    FileIO.read_pickle(reader.get_path('runs/0/training_task.pkl'))
except:
    print(traceback.format_exc())
```

    Traceback (most recent call last):
      File "/tmp/ipykernel_4552/225246354.py", line 3, in <module>
        FileIO.read_pickle(reader.get_path('runs/0/training_task.pkl'))
      File "/home/yura/anaconda3/envs/tg/lib/python3.8/site-packages/yo_fluq_ds/_misc/io.py", line 17, in read_pickle
        return pickle.load(file)
    ModuleNotFoundError: No module named 'titanic___20220627_214109_80c8c40453a544cc99e69cd531aaeb71'
    


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





# 5. Analytics tools (tg.common.analysis)

## Overview

`tg.common.analysis` is a module with some helpful analytics tools, mostly around the concept of statistical significance, with the aim to make it easier to use this concept and visualize the results in the report. Currently, very few features are implemented, but those are extremely useful for the meaningful reports.

## `percentile_confint`


As a model example, we will use Titanic dataset again, particularly around the question "which features are most useful to predict the outcome". We have demonstrated this technique in `tg.common.ml.single_frame_training`, but it's also implemented as ready to use solution in `tg.common.analytics`.


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




```python
from tg.common.analysis import FeatureSignificance
from tg.common import Logger

Logger.disable()

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
significance = FeatureSignificance.for_classification_task(
    df = df,
    features = features,
    label = 'Survived',
    folds_count = 200
)
```


      0%|          | 0/200 [00:00<?, ?it/s]



```python
significance.head()
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
      <th>Embarked_S</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_NULL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.861870</td>
      <td>-0.571329</td>
      <td>-0.430517</td>
      <td>-0.095062</td>
      <td>0.062456</td>
      <td>-0.301913</td>
      <td>-1.334796</td>
      <td>1.334920</td>
      <td>-0.418228</td>
      <td>0.062635</td>
      <td>0.186314</td>
      <td>0.169403</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.985657</td>
      <td>-0.617876</td>
      <td>-0.347725</td>
      <td>0.038203</td>
      <td>-0.034456</td>
      <td>-0.428030</td>
      <td>-1.381279</td>
      <td>1.381317</td>
      <td>-0.427300</td>
      <td>0.050905</td>
      <td>0.243914</td>
      <td>0.132519</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.954792</td>
      <td>-0.669994</td>
      <td>-0.404068</td>
      <td>-0.043028</td>
      <td>-0.005523</td>
      <td>-0.355824</td>
      <td>-1.395271</td>
      <td>1.395354</td>
      <td>-0.373559</td>
      <td>0.122871</td>
      <td>0.050387</td>
      <td>0.200385</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.945720</td>
      <td>-0.658512</td>
      <td>-0.478880</td>
      <td>-0.050786</td>
      <td>0.294873</td>
      <td>-0.200040</td>
      <td>-1.372717</td>
      <td>1.372751</td>
      <td>-0.461177</td>
      <td>0.151009</td>
      <td>0.210827</td>
      <td>0.099375</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.877177</td>
      <td>-0.507637</td>
      <td>-0.273289</td>
      <td>-0.022627</td>
      <td>0.099942</td>
      <td>-0.472386</td>
      <td>-1.226026</td>
      <td>1.226146</td>
      <td>-0.434261</td>
      <td>0.054348</td>
      <td>0.209524</td>
      <td>0.170508</td>
    </tr>
  </tbody>
</table>
</div>




```python
from matplotlib import pyplot as plt
from seaborn import violinplot

_, ax = plt.subplots(1,1,figsize=(20,5))
violinplot(data=significance, ax=ax)
pass
```


    
![png](README_images/tg.common.analysis_output_6_0.png?raw=true)
    


We see that some of the feature are reallyfar from zero (like `Sex_male` and `Sex_female`), but some others are near and it's hard to say, if there are significant. To answer this question, we can build a confidence intervals:


```python
sdf = significance.unstack().to_frame().reset_index()
sdf.columns = ['feature','experiment','value']
sdf.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>experiment</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pclass</td>
      <td>0</td>
      <td>-0.861870</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pclass</td>
      <td>1</td>
      <td>-0.985657</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pclass</td>
      <td>2</td>
      <td>-0.954792</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pclass</td>
      <td>3</td>
      <td>-0.945720</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass</td>
      <td>4</td>
      <td>-0.877177</td>
    </tr>
  </tbody>
</table>
</div>




```python
from tg.common.analysis import Aggregators
sdf1 = sdf.groupby('feature').value.feed(Aggregators.percentile_confint(pValue=0.9)).reset_index()
sdf1.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>value_lower</th>
      <th>value_upper</th>
      <th>value_value</th>
      <th>value_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>-0.670278</td>
      <td>-0.440688</td>
      <td>-0.555483</td>
      <td>0.114795</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age_missing</td>
      <td>-0.482734</td>
      <td>-0.031772</td>
      <td>-0.257253</td>
      <td>0.225481</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Embarked_C</td>
      <td>-0.080599</td>
      <td>0.239275</td>
      <td>0.079338</td>
      <td>0.159937</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Embarked_NULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Embarked_Q</td>
      <td>-0.080665</td>
      <td>0.360177</td>
      <td>0.139756</td>
      <td>0.220421</td>
    </tr>
  </tbody>
</table>
</div>



This aggregator is essentially doing what `mean` or `std` functions do, but for statistical significance. 

Now, we can use `grbar_plot` for visualization:


```python
from tg.common.analysis import grbar_plot

grbar_plot(
    sdf1.loc[sdf1.value_upper*sdf1.value_lower>0].sort_values('value_upper'), 
    value_column='value_value', 
    error_column='value_error', 
    group_column='feature', orient='h'
)
```




    <AxesSubplot:xlabel='value_value', ylabel='feature'>




    
![png](README_images/tg.common.analysis_output_11_1.png?raw=true)
    


## `proportion_confint`

Aside from `percentile_confint`, which builds a confidence interval by simply computing the borders in which the proper amount of values fall, there is `proportion_confint` that uses the formula for Bernoulli distribution.


```python
df.Survived.feed(Aggregators.proportion_confint())
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived_lower</th>
      <th>Survived_upper</th>
      <th>Survived_value</th>
      <th>Survived_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.351906</td>
      <td>0.415771</td>
      <td>0.383838</td>
      <td>0.031932</td>
    </tr>
  </tbody>
</table>
</div>



As any Aggregator, it can be applied, e.g., to groups:


```python
qdf = df.groupby(['Sex','Embarked']).Survived.feed(Aggregators.proportion_confint()).reset_index()
qdf
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Embarked</th>
      <th>Survived_lower</th>
      <th>Survived_upper</th>
      <th>Survived_value</th>
      <th>Survived_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>C</td>
      <td>0.801294</td>
      <td>0.952130</td>
      <td>0.876712</td>
      <td>0.075418</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>Q</td>
      <td>0.608552</td>
      <td>0.891448</td>
      <td>0.750000</td>
      <td>0.141448</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>S</td>
      <td>0.626014</td>
      <td>0.753296</td>
      <td>0.689655</td>
      <td>0.063641</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>C</td>
      <td>0.212658</td>
      <td>0.397868</td>
      <td>0.305263</td>
      <td>0.092605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>Q</td>
      <td>0.000000</td>
      <td>0.152883</td>
      <td>0.076441</td>
      <td>0.076441</td>
    </tr>
    <tr>
      <th>5</th>
      <td>male</td>
      <td>S</td>
      <td>0.139172</td>
      <td>0.210034</td>
      <td>0.174603</td>
      <td>0.035431</td>
    </tr>
  </tbody>
</table>
</div>



`grbar_plot` can also use two variables for drawing:


```python
grbar_plot(
    qdf, 
    group_column='Sex', 
    color_column='Embarked', 
    value_column='Survived_value', 
    error_column='Survived_error')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f07ca52e730>




    
![png](README_images/tg.common.analysis_output_18_1.png?raw=true)
    


## Bootstrap

Let's explore if there is a significant difference between fares for men and women. 


```python
df.groupby('Sex').Fare.mean()
```




    Sex
    female    44.479818
    male      25.523893
    Name: Fare, dtype: float64



Well, maybe, but is this difference significant? 

In this case, it's hard to use math: most of the cases are for normal distribution, but `Fare` is not distributed normally.


```python
df.Fare.hist()
```




    <AxesSubplot:>




    
![png](README_images/tg.common.analysis_output_22_1.png?raw=true)
    


We can use [bootstraping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) for this task. To do so, first, we need a function that computes the value of interest as a dataframe:


```python
def compute(df):
    return df.groupby('Sex').Fare.mean().to_frame().transpose().reset_index()
    
compute(df)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sex</th>
      <th>index</th>
      <th>female</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fare</td>
      <td>44.479818</td>
      <td>25.523893</td>
    </tr>
  </tbody>
</table>
</div>




```python
from tg.common.analysis import Bootstrap

bst = Bootstrap(df = df, method = compute)
rdf = bst.run(N=1000)
```


      0%|          | 0/1000 [00:00<?, ?it/s]



```python
rdf.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>female</th>
      <th>male</th>
      <th>iteration</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fare</td>
      <td>43.462195</td>
      <td>26.987083</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fare</td>
      <td>41.431770</td>
      <td>25.906982</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fare</td>
      <td>40.536302</td>
      <td>27.590799</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fare</td>
      <td>44.421407</td>
      <td>25.357936</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fare</td>
      <td>42.063531</td>
      <td>24.546184</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



So, `Bootstrap` took subsamples of our task and computed means for each of them, creating the distribution of means:


```python
rdf.female.hist()
rdf.male.hist()
```




    <AxesSubplot:>




    
![png](README_images/tg.common.analysis_output_28_1.png?raw=true)
    


Those are normal, and bootsraping guarantees it (providing that mean value exists and enough samples are taken). 

So, we can use confidence intervals for normal distribution:


```python
rdf[['female','male']].feed(Aggregators.normal_confint())
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>female_lower</th>
      <th>female_upper</th>
      <th>female_value</th>
      <th>female_error</th>
      <th>male_lower</th>
      <th>male_upper</th>
      <th>male_value</th>
      <th>male_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39.053246</td>
      <td>50.985264</td>
      <td>45.019255</td>
      <td>5.966009</td>
      <td>22.117442</td>
      <td>29.279721</td>
      <td>25.698581</td>
      <td>3.581139</td>
    </tr>
  </tbody>
</table>
</div>



When data are computed this way, it's not really easy to visualize, so:


```python
rdf1 = rdf[['female','male']].unstack().to_frame().reset_index()
rdf1.columns=['sex','iteration','fare']

grbar_plot(
    rdf1.groupby('sex').fare.feed(Aggregators.normal_confint()).reset_index(),
    value_column='fare_value',
    error_column='fare_error',
    group_column='sex'
)
```




    <AxesSubplot:xlabel='sex', ylabel='fare_value'>




    
![png](README_images/tg.common.analysis_output_32_1.png?raw=true)
    


The confidence intervals do not intersect (which, honestly, was quite visible from histogram), thus, the difference is significant.
