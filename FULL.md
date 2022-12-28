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

    2022-12-28 14:20:03.949207 INFO: Message with default logger



```python
Logger.initialize_kibana()
Logger.info('Message with Kibana logger')
```

    {"@timestamp": "2022-12-28T13:20:03.955688+00:00", "message": "Message with Kibana logger", "levelname": "INFO", "logger": "tg", "path": "/tmp/ipykernel_16175/2907404717.py", "path_line": 2}


As said before, you may define a custom session keys:


```python
Logger.push_keys(test_key='test')
Logger.info('Message with a key')
Logger.clear_keys()
Logger.info('Message without a key')
```

    {"@timestamp": "2022-12-28T13:20:03.961362+00:00", "message": "Message with a key", "levelname": "INFO", "logger": "tg", "path": "/tmp/ipykernel_16175/71300885.py", "path_line": 2, "test_key": "test"}
    {"@timestamp": "2022-12-28T13:20:03.962184+00:00", "message": "Message without a key", "levelname": "INFO", "logger": "tg", "path": "/tmp/ipykernel_16175/71300885.py", "path_line": 4}


If exception information is available, it will be put in the keys:


```python
try:
    raise ValueError('Error')
except: 
    Logger.error('Error')
```

    {"@timestamp": "2022-12-28T13:20:03.968112+00:00", "message": "Error", "levelname": "ERROR", "logger": "tg", "path": "/tmp/ipykernel_16175/1975102656.py", "path_line": 4, "exception_type": "<class 'ValueError'>", "exception_value": "Error", "exception_details": "Traceback (most recent call last):\n  File \"/tmp/ipykernel_16175/1975102656.py\", line 2, in <module>\n    raise ValueError('Error')\nValueError: Error\n"}


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




    ('ced7a461-b52d-47c9-8c4a-146435d966c4',
     'ced7a461-b52d-47c9-8c4a-146435d966c4',
     '93139005-03c6-4657-a411-0794eeaa5a43')



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

    2022-12-28 14:20:12.619902 WARNING: Missing field in FieldGetter





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

    2022-12-28 14:20:12.636041 WARNING: Missing field in FieldGetter





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
     'processed': datetime.datetime(2022, 12, 28, 14, 20, 12, 636841)}



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
           "length": "<function get_length at 0x7f0d5f433d30>",
           "title": "<function get_title at 0x7f0d5f433790>"
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
       1: {'length': '<function get_length at 0x7f0d5f433d30>',
        'title': '<function get_title at 0x7f0d5f433790>'}},
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

    2022-12-28 14:20:12.715036 WARNING: Missing field in FieldGetter
    2022-12-28 14:20:12.716139 WARNING: Missing field in FieldGetter
    2022-12-28 14:20:12.717215 WARNING: Missing field in FieldGetter





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
      <td>2022-12-28 14:20:12.715806</td>
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
      <td>2022-12-28 14:20:12.716881</td>
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
      <td>2022-12-28 14:20:12.717855</td>
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

    2022-12-28 14:20:17.274079 INFO: Featurization Job job at version v1 has started
    2022-12-28 14:20:17.276747 INFO: Fetching data
    2022-12-28 14:20:17.489488 INFO: Data fetched, finalizing
    2022-12-28 14:20:17.563401 INFO: Uploading data
    2022-12-28 14:20:17.568566 INFO: Featurization job completed


Some notes: 

* `DataFrameFeaturizer`: When used in this way, it just applies `row_selector` to each data object from `source` and collects the results into pandas dataframes
* If no `location` is provided, the folder will be created automatically in the `Loc.temp_path` folder. Usually we don't care where the intermediate files are stored, as syncer takes care of them automatically.
* `MemoryFileSyncer`. The job creates files locally (in the `location` folder), and the uploads them to the remote destination. For demonstration purposes, we will "upload" data in the memory. `tg.common` also contains `S3FileSyncer` that syncs the files with `S3`. Interfaces for other storages may be written, deriving from `FileSyncer`. Essentialy, the meaning of `FileSyncer` is a connection between a specific location on the local disk and the location somewhere else. When calling `upload` or `download` methods, the class assures the same content of given files/folders.


The resulting files can be viewed in the following way:


```python
list(mem.cache)
```




    ['passengers/2b993afd-6784-4fa0-9148-254eddc20cb1.parquet']



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

    2022-12-28 14:20:17.667424 INFO: Featurization Job job at version v1 has started
    2022-12-28 14:20:17.670882 INFO: Fetching data
    2022-12-28 14:20:17.750987 INFO: Data fetched, finalizing
    2022-12-28 14:20:17.757166 INFO: Uploading data
    2022-12-28 14:20:17.758679 INFO: Featurization job completed





    ['passengers/71b6b93d-1b21-41e4-8a39-24ccbcde5848.parquet',
     'passengers/babdf725-f27a-4c1e-9830-12577819e902.parquet',
     'passengers/0c425f1f-5cb1-4020-b807-a93126da3116.parquet',
     'passengers/fdc573ca-606e-4586-a46d-fe5221dc8246.parquet']




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

    2022-12-28 14:20:17.781484 INFO: Featurization Job job at version v1 has started
    2022-12-28 14:20:17.783203 INFO: Fetching data
    2022-12-28 14:20:17.839942 INFO: Data fetched, finalizing
    2022-12-28 14:20:17.847316 INFO: Uploading data
    2022-12-28 14:20:17.848311 INFO: Featurization job completed





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

    2022-12-28 14:20:17.869530 INFO: Featurization Job job at version v1 has started
    2022-12-28 14:20:17.872444 INFO: Fetching data
    2022-12-28 14:20:17.924502 INFO: Data fetched, finalizing
    2022-12-28 14:20:17.935663 INFO: Uploading data
    2022-12-28 14:20:17.936865 INFO: Featurization job completed





    ['cabins/6c34a07d-4ac4-41ce-bac7-481a23701a0e.parquet',
     'passengers/aacae0ab-0ba2-4d67-ab02-d569455f6100.parquet']



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

    2022-12-28 14:20:20.265386 INFO: Starting lesvik job test_featurization_job, version v1
    2022-12-28 14:20:20.266654 INFO: Additional settings limit NONE, reporting NONE
    2022-12-28 14:20:20.267648 INFO: 0 previous revisions are found
    2022-12-28 14:20:20.268154 INFO: Running with id 0 at 2020-01-01 00:00:00, revision is MAJOR
    2022-12-28 14:20:20.268623 INFO: Featurization Job test_featurization_job at version v1 has started
    2022-12-28 14:20:20.274809 INFO: Fetching data
    2022-12-28 14:20:20.331211 INFO: Data fetched, finalizing
    2022-12-28 14:20:20.342725 INFO: Uploading data
    2022-12-28 14:20:20.343953 INFO: Featurization job completed
    2022-12-28 14:20:20.344573 INFO: 891 were processed
    2022-12-28 14:20:20.344993 INFO: Uploading new description
    2022-12-28 14:20:20.348811 INFO: Job finished
    2022-12-28 14:20:20.349338 INFO: Starting lesvik job test_featurization_job, version v1
    2022-12-28 14:20:20.349793 INFO: Additional settings limit NONE, reporting NONE
    2022-12-28 14:20:20.353792 INFO: 1 previous revisions are found
    2022-12-28 14:20:20.354260 INFO: Running with id 2 at 2020-01-03 00:00:00, revision is MINOR
    2022-12-28 14:20:20.354596 INFO: Featurization Job test_featurization_job at version v1 has started
    2022-12-28 14:20:20.359414 INFO: Fetching data
    2022-12-28 14:20:20.369041 INFO: Data fetched, finalizing
    2022-12-28 14:20:20.374701 INFO: Uploading data
    2022-12-28 14:20:20.375694 INFO: Featurization job completed
    2022-12-28 14:20:20.376128 INFO: 168 were processed
    2022-12-28 14:20:20.376520 INFO: Uploading new description
    2022-12-28 14:20:20.379880 INFO: Job finished
    2022-12-28 14:20:20.380402 INFO: Starting lesvik job test_featurization_job, version v1
    2022-12-28 14:20:20.380797 INFO: Additional settings limit NONE, reporting NONE
    2022-12-28 14:20:20.384814 INFO: 2 previous revisions are found
    2022-12-28 14:20:20.385369 INFO: Running with id 4 at 2020-01-05 00:00:00, revision is MINOR
    2022-12-28 14:20:20.385709 INFO: Featurization Job test_featurization_job at version v1 has started
    2022-12-28 14:20:20.390244 INFO: Fetching data
    2022-12-28 14:20:20.395039 INFO: Data fetched, finalizing
    2022-12-28 14:20:20.401771 INFO: Uploading data
    2022-12-28 14:20:20.402909 INFO: Featurization job completed
    2022-12-28 14:20:20.404464 INFO: 77 were processed
    2022-12-28 14:20:20.406484 INFO: Uploading new description
    2022-12-28 14:20:20.410661 INFO: Job finished
    2022-12-28 14:20:20.411241 INFO: Starting lesvik job test_featurization_job, version v1
    2022-12-28 14:20:20.411682 INFO: Additional settings limit NONE, reporting NONE
    2022-12-28 14:20:20.421033 INFO: 3 previous revisions are found
    2022-12-28 14:20:20.422118 INFO: Running with id 6 at 2020-01-07 00:00:00, revision is MINOR
    2022-12-28 14:20:20.422622 INFO: Featurization Job test_featurization_job at version v1 has started
    2022-12-28 14:20:20.427384 INFO: Fetching data
    2022-12-28 14:20:20.474042 INFO: Data fetched, finalizing
    2022-12-28 14:20:20.482652 INFO: Uploading data
    2022-12-28 14:20:20.483651 INFO: Featurization job completed
    2022-12-28 14:20:20.484254 INFO: 644 were processed
    2022-12-28 14:20:20.484739 INFO: Uploading new description
    2022-12-28 14:20:20.488453 INFO: Job finished


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
      <td>9cee2b45-6757-4d05-a965-23bbe7156beb.parquet</td>
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
      <td>8877c73f-43db-49d6-8436-0088ffdc6a8b.parquet</td>
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
      <td>2d6d2392-9761-4822-b67e-53e00b81d60c.parquet</td>
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
      <td>032f775d-7f9f-4b40-9bb7-2cd72519d590.parquet</td>
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
      <td>e94bf29e-9cc4-42ef-8f06-0315a822d532.parquet</td>
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
      <td>2bd928b5-f354-4958-9410-746774ea2d0e.parquet</td>
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
      <td>20ebdf31-5b8b-4f4d-a585-7f7eb73c5807.parquet</td>
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
      <td>29a15853-1fe2-49e6-bcbc-7a28bf4573ed.parquet</td>
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
      <td>e7badd8a-e8ee-4a7d-9521-f5345fbc2c34.parquet</td>
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
      <td>fb9582ff-1e13-4389-b718-696bc88c756b.parquet</td>
      <td>723.0</td>
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
      <td>5587e92b-9d71-4f5e-9502-30737a6f2cdb.parquet</td>
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
      <td>325cbab0-1d3e-44ae-86b5-8973677d50b2.parquet</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2020-01-05 01:00:00</td>
      <td>True</td>
      <td></td>
      <td>bdeb970a-22d3-4036-a0ea-0ae319ef080b.parquet</td>
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
      <td>09fb89bd-70cb-4e06-9b89-d7c2ef982857.parquet</td>
      <td>646.0</td>
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
      <td>c29aa577-79c7-4d1c-9442-89ecf3cac2f6.parquet</td>
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
      <td>db3d8fa6-1cb9-4de8-a2a5-815b9ae804fd.parquet</td>
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
      <td>4c6683a0-ff24-4b17-aff3-5051d66529f4.parquet</td>
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
      <td>5fc276af-15c0-4344-a4c7-a166329483d8.parquet</td>
      <td>NaN</td>
      <td>168.0</td>
      <td>NaN</td>
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

The Data Cleaning phase is applicable to the stage when we already have our dataset as a tidy dataframe. This dataframe may not, however, be immediately suitable for machine leaning as it may contain:

* missing continuous values
* non-normalized continuous values
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



`tfac` is a data transformer in the sense of `sklearn`, it has the `fit`, `transform` and `fit_transform` methods.

The default solution:
  * automatically determines if the feature is continuous or categorical
  * performs normalisation and imputation to continous variables, as well as adds the missing indicator
  * for categorical variables:
    * applies one-hot encoding
    * Converts None variable to a string NONE
    * limits the amount of values per feature, placing least-popular values in `OTHER` column (this is crucial, e.g., for decision trees)


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

test_df = pd.DataFrame([dict(Survived=0, Age=30, SibSp=0, Fare=None, Parch=0, PClass=0)]).astype(float)
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

    2022-12-28 14:20:25.225323 WARNING: Missing column in ContinuousTransformer


    Traceback (most recent call last):
      File "/tmp/ipykernel_16374/777428497.py", line 14, in <module>
        tr.transform(test_df)
      File "/home/yura/Desktop/repos/appalack-ml/tg/common/ml/dft/architecture.py", line 48, in transform
        for res in transformer.transform(df):
      File "/home/yura/Desktop/repos/appalack-ml/tg/common/ml/dft/column_transformers.py", line 90, in transform
        missing = self.missing_indicator.transform(subdf)
      File "/home/yura/anaconda3/envs/ap/lib/python3.8/site-packages/sklearn/impute/_base.py", line 885, in transform
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

    2022-12-28 14:20:25.256829 WARNING: Missing column in ContinuousTransformer
    2022-12-28 14:20:25.265063 WARNING: Unexpected None in MissingIndicatorWithReporting
    2022-12-28 14:20:25.265807 WARNING: Unexpected None in MissingIndicatorWithReporting





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
      <td>-0.473674</td>
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

    {"@timestamp": "2022-12-28T13:20:25.278713+00:00", "message": "Missing column in ContinuousTransformer", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/appalack-ml/tg/common/ml/dft/column_transformers.py", "path_line": 75, "column": "Pclass"}
    {"@timestamp": "2022-12-28T13:20:25.290459+00:00", "message": "Unexpected None in MissingIndicatorWithReporting", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/appalack-ml/tg/common/ml/dft/miscellaneous.py", "path_line": 36, "column": "Pclass"}
    {"@timestamp": "2022-12-28T13:20:25.292595+00:00", "message": "Unexpected None in MissingIndicatorWithReporting", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/appalack-ml/tg/common/ml/dft/miscellaneous.py", "path_line": 36, "column": "Fare"}





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
      <td>-0.473674</td>
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

    {"@timestamp": "2022-12-28T13:20:25.390231+00:00", "message": "Unexpected value in MostPopularStrategy", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/appalack-ml/tg/common/ml/dft/column_transformers.py", "path_line": 122, "column": "Embarked", "value": "NONE"}





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

... That's also fine. TG is designed to make the life easier, not worse. We offer the SOLID implementation for two wide-spread training scenarios, and we believe that this is a generally better way. But if you are uncomfortable with the SOLID approach to machine learning, or your process is so specific that it does not fit into both scenarios we have implemented, you always have the following option:

* Inherit from `AbstractTrainingTask`
* Implement `run_with_environment` method and write the code in any way you see appropriate
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


    2022-12-28 14:20:29.533243 INFO: Starting stage 1/1
    2022-12-28 14:20:29.726526 INFO: Completed stage 1/1


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
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7ff71a3ee9d0>),
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
      <td>0.870149</td>
      <td>0.998156</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.833078</td>
      <td>0.998982</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.833681</td>
      <td>0.997666</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.817806</td>
      <td>0.999574</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.884988</td>
      <td>0.997668</td>
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
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7ff718b93190>),
                    ('ColumnNamesKeeperAfterTransformation', ColumnNamesKeeper()),
                    ('Model',
                     Pipeline(steps=[('CategoricalVariablesSetter',
                                      <tg.common.ml.single_frame_training.model_provider.CatBoostWrap object at 0x7ff71a3bc3d0>),
                                     ('Model',
                                      <catboost.core.CatBoostClassifier object at 0x7ff71a3bc2e0>)]))])



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
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7ff70404b730>),
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
                     <tg.common.ml.dft.transform_factory.DataFrameTransformerFactory object at 0x7ff73d6dda60>),
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
    1     982
    2    1994
    3    2967
    4    4057
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



The most simple case is when data extracted from index itself, which is the case for the label:


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
    extractors = [feature_extractor, label_extractor],
    batching_strategy = bt.PriorityRandomBatcherStrategy('priority')
)
```

Let's take a look at the batch produced


```python
batch_size = 50
batch = batcher.fit_extract(batch_size, ibundle_fixed)
batch
```




    <tg.common.ml.batched_training.data_bundle.IndexedDataBundle at 0x7f8c24b0fb50>



So, batch is of type `DataBundle`, and contains index, features, and labels. 

The batch is balanced


```python
batch['labels'].groupby('Survived').size()
```




    Survived
    0.0    30
    1.0    17
    dtype: int64



**Note:** due to the technical reasons, `PlainExtractor`, as well as other extractors, do not support extraction in case when the output of `BatchingStrategy` contains duplicated rows. This is why by default, `PriorityRandomBatcherStrategy` deduplicates them, and therefore cannot return more rows that there are in the bundle. We consider fixing this issue in the future releases, but since the datasets are normaly (much) bigger than batches, not with the high priority.


```python
test_batcher = bt.Batcher(
    extractors = [feature_extractor, label_extractor],
    batching_strategy = bt.PriorityRandomBatcherStrategy('priority')
)
test_batch = test_batcher.fit_extract(batch_size, ibundle_fixed)
test_batch['labels'].groupby('Survived').size()
```




    Survived
    0.0    22
    1.0    27
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

In `tg.common.ml.batched_training.factories` there is a generic definition for such `ModelHandler` that we will cover in the corresponding demo. Here, we will define `ModelHandler` from scratch, to demonstrate its logic.



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
task.settings.batch_size=100

result = task.run(ibundle_fixed)
```

    2022-12-28 14:21:10.014697 INFO: Training starts. Info: {}
    2022-12-28 14:21:10.016047 INFO: Ensuring/loading bundle. Bundle before:
    <tg.common.ml.batched_training.data_bundle.IndexedDataBundle object at 0x7f8c4038d640>
    2022-12-28 14:21:10.016617 INFO: Bundle loaded
    {'index': {'shape': (891, 5), 'index_name': 'PassengerId', 'columns': ['Name', 'Ticket', 'Cabin', 'Survived', 'priority'], 'index': [1, 2, 3, 4, 5, '...']}, 'passengers': {'shape': (891, 4), 'index_name': 'Name', 'columns': ['Sex', 'Age', 'SibSp', 'Parch'], 'index': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath (Lily May Peel)', 'Allen, Mr. William Henry', '...']}, 'tickets': {'shape': (681, 3), 'index_name': 'Ticket', 'columns': ['Pclass', 'Fare', 'Embarked'], 'index': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450', '...']}}
    2022-12-28 14:21:10.017190 INFO: Index frame is set to index, shape is (891, 5)
    2022-12-28 14:21:10.017786 INFO: Skipping late initialization
    2022-12-28 14:21:10.018319 INFO: Preprocessing bundle by batcher
    2022-12-28 14:21:10.023948 INFO: Splits: train 712, test 179, display 143
    2022-12-28 14:21:10.024546 INFO: New training. Instantiating the system
    2022-12-28 14:21:10.025671 INFO: Fitting the transformers
    2022-12-28 14:21:10.073160 INFO: Instantiating model
    2022-12-28 14:21:10.074611 INFO: Initialization completed
    2022-12-28 14:21:10.075486 INFO: Epoch 0 of 1
    2022-12-28 14:21:10.075927 INFO: Training: 0/8
    2022-12-28 14:21:10.103902 INFO: Training: 1/8
    2022-12-28 14:21:10.128179 INFO: Training: 2/8
    2022-12-28 14:21:10.156215 INFO: Training: 3/8
    2022-12-28 14:21:10.180618 INFO: Training: 4/8
    2022-12-28 14:21:10.205398 INFO: Training: 5/8
    2022-12-28 14:21:10.230677 INFO: Training: 6/8
    2022-12-28 14:21:10.257123 INFO: Training: 7/8
    2022-12-28 14:21:10.282654 INFO: test: 0/2
    2022-12-28 14:21:10.308441 INFO: test: 1/2
    2022-12-28 14:21:10.334269 INFO: display: 0/2
    2022-12-28 14:21:10.358629 INFO: display: 1/2
    2022-12-28 14:21:10.387787 INFO: ###roc_auc_score_test:0.4130434782608696
    2022-12-28 14:21:10.388229 INFO: ###roc_auc_score_display:0.43268817204301074
    2022-12-28 14:21:10.388711 INFO: ###loss:0.2608289998024702
    2022-12-28 14:21:10.389460 INFO: ###iteration:0


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
    settings = bt.TrainingSettings(epoch_count=10, batch_size=100),
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
      <td>0.385275</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.664146</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.462648</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.632161</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.384694</td>
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
batch, temp_data = task.generate_sample_batch_and_temp_data(ibundle_fixed)
```

This batch then can be used on different levels to debug network and handler:


```python
task.model_handler.network(batch['features'])[:5]
```




    tensor([[0.4334],
            [0.4201],
            [0.4227],
            [0.4231],
            [0.4233]], grad_fn=<SliceBackward>)




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
      <th>874</th>
      <td>0.433400</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>833</th>
      <td>0.420091</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>105</th>
      <td>0.422657</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>612</th>
      <td>0.423145</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>284</th>
      <td>0.423349</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Temp data also contains lots of the information, useful for debugging, such as splits


```python
temp_data.__dict__
```




    {'original_ibundle': <tg.common.ml.batched_training.data_bundle.IndexedDataBundle at 0x7f8c4038d640>,
     'env': <tg.common.ml.training_core.arch.TrainingEnvironment at 0x7f8bac086a90>,
     'split': <tg.common.ml.training_core.splitter.DataFrameSplit at 0x7f8ba8f45a30>,
     'first_iteration': 0,
     'iteration': 0,
     'losses': [],
     'epoch_begins_at': None,
     'train_bundle': None,
     'result': None,
     'batch': None,
     'mini_batch_indices': None,
     'mini_batch': None}



### Debug mode

`BatchingTrainingTask` has a `debug` argument, which forces the task to keep the intermediate data as a field of the class (`temp_data` from the previous section). **Never** do it in production, as the intermediate data also contain the bundle, so pickling the task (which is an artefact of the training) will be impossible with any real data. 

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
    0.0    439
    1.0    273
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





# 3.3.1. Batched training with torch and factories (tg.common.ml.batched_training.factories)

Thousands of training processes have shown `BatchedTrainingTask` is a reliable and effective way of training neural networks. 

However, we experienced some problems with training networks of different architectures on data with complex structures (such as contextual data, see `tg.common.ml.batched_training.context`). These problems are conceptual: the approach of `BatchedTrainingTask` is that the training is a SOLID object entirely configurable by components; however, in the reality the configuration can only be achieved by extensive coding in `ModelHandler` and `lazy_initialization`. 

`tg.common.ml.batched_training.factories` addresses the problem, subclassing and adjusting `BatchedTrainingTask` and `ModelHandler` for this scenario, as well as adding some additional classes for network creation.

## Binary classification task

We will work on binary classification task from standard sklearn datasets. First, we need to translate it into bundle:


```python
from yo_fluq_ds import *
from sklearn import datasets
import pandas as pd
from tg.common.ml import batched_training as bt

def get_binary_classification_bundle():
    ds = datasets.load_breast_cancer()
    features = pd.DataFrame(ds['data'], columns=ds['feature_names'])
    df = pd.DataFrame(ds['target'], columns=['label'])
    df['split'] = bt.train_display_test_split(df, 0.2, 0.2, 'label')
    bundle = bt.DataBundle(index=df, features=features)
    return bundle

get_binary_classification_bundle().features.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
get_binary_classification_bundle().index.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>display</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>train</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>train</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>



We see that index label contains both label and split. This is a recommended way of splitting for `TorchTrainingTask`: set the split in bundle. The reason for this is that we can't really use several splits in batched training the way we did for single frme task, so the whole architecture of splits becomes unusable. Also, when comparing many networks against each other, it's good to ensure that they train on exactly same data.

Let's define the extractors. This part didn't change in comparison with `tg.common.ml.batched_training`


```python
from tg.common.ml.batched_training import factories as btf
from tg.common.ml import dft

def get_feature_extractor():
    feature_extractor = (bt.PlainExtractor
                 .build('features')
                 .index('features')
                 .apply(transformer = dft.DataFrameTransformerFactory.default_factory())
                )
    return feature_extractor
    
def get_binary_label_extractor():
    label_extractor = (bt.PlainExtractor
                   .build(btf.Conventions.LabelFrame)
                   .index()
                   .apply(take_columns=['label'], transformer=None)
                  )
    return label_extractor

def test_extractor(extractor, bundle):
    extractor.fit(bundle)
    return extractor.extract(bundle)

db = get_binary_classification_bundle()
idb = bt.IndexedDataBundle(db.index, db)
test_extractor( get_feature_extractor(), idb).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.097064</td>
      <td>-2.073335</td>
      <td>1.269934</td>
      <td>0.984375</td>
      <td>1.568466</td>
      <td>3.283515</td>
      <td>2.652874</td>
      <td>2.532475</td>
      <td>2.217515</td>
      <td>2.255747</td>
      <td>...</td>
      <td>1.886690</td>
      <td>-1.359293</td>
      <td>2.303601</td>
      <td>2.001237</td>
      <td>1.307686</td>
      <td>2.616665</td>
      <td>2.109526</td>
      <td>2.296076</td>
      <td>2.750622</td>
      <td>1.937015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.829821</td>
      <td>-0.353632</td>
      <td>1.685955</td>
      <td>1.908708</td>
      <td>-0.826962</td>
      <td>-0.487072</td>
      <td>-0.023846</td>
      <td>0.548144</td>
      <td>0.001392</td>
      <td>-0.868652</td>
      <td>...</td>
      <td>1.805927</td>
      <td>-0.369203</td>
      <td>1.535126</td>
      <td>1.890489</td>
      <td>-0.375612</td>
      <td>-0.430444</td>
      <td>-0.146749</td>
      <td>1.087084</td>
      <td>-0.243890</td>
      <td>0.281190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.579888</td>
      <td>0.456187</td>
      <td>1.566503</td>
      <td>1.558884</td>
      <td>0.942210</td>
      <td>1.052926</td>
      <td>1.363478</td>
      <td>2.037231</td>
      <td>0.939685</td>
      <td>-0.398008</td>
      <td>...</td>
      <td>1.511870</td>
      <td>-0.023974</td>
      <td>1.347475</td>
      <td>1.456285</td>
      <td>0.527407</td>
      <td>1.082932</td>
      <td>0.854974</td>
      <td>1.955000</td>
      <td>1.152255</td>
      <td>0.201391</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.768909</td>
      <td>0.253732</td>
      <td>-0.592687</td>
      <td>-0.764464</td>
      <td>3.283553</td>
      <td>3.402909</td>
      <td>1.915897</td>
      <td>1.451707</td>
      <td>2.867383</td>
      <td>4.910919</td>
      <td>...</td>
      <td>-0.281464</td>
      <td>0.133984</td>
      <td>-0.249939</td>
      <td>-0.550021</td>
      <td>3.394275</td>
      <td>3.893397</td>
      <td>1.989588</td>
      <td>2.175786</td>
      <td>6.046041</td>
      <td>4.935010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.750297</td>
      <td>-1.151816</td>
      <td>1.776573</td>
      <td>1.826229</td>
      <td>0.280372</td>
      <td>0.539340</td>
      <td>1.371011</td>
      <td>1.428493</td>
      <td>-0.009560</td>
      <td>-0.562450</td>
      <td>...</td>
      <td>1.298575</td>
      <td>-1.466770</td>
      <td>1.338539</td>
      <td>1.220724</td>
      <td>0.220556</td>
      <td>-0.313395</td>
      <td>0.613179</td>
      <td>0.729259</td>
      <td>-0.868353</td>
      <td>-0.397100</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
test_extractor(get_binary_label_extractor(), idb).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



In `tg.common.ml.batched_training.factories`, `TorchModelHandler` class is defined. A short reminder: `ModelHandler` class should handle initialization, training and prediction of the model. The `TorchModelHandler` addresses the last two in a generic way, and completely outsources the inialization to three entities:
* Network factory
* Optimizer constructor
* Loss constructor

Let's cover the first one. In general, network factory is an arbitrary function that accepts one batch and generates the network. This gives us the opportunity to adjust the network to the input data: the shape of the input data is determined after the extractors are fitted, which is very late in the initialization process.


```python
from sklearn.metrics import roc_auc_score
from tg.common import Logger
from functools import partial
import torch

Logger.disable()

class ClassificationNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassificationNetwork, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        X = input['features']
        X = torch.tensor(X.astype(float).values).float()
        X = self.hidden(X)
        X = torch.sigmoid(X)
        X = self.output(X)
        X = torch.sigmoid(X)
        return X
    
def create_factory(hidden_size):
    return lambda sample: ClassificationNetwork(
        sample['features'].shape[1], 
        hidden_size, 
        sample[btf.Conventions.LabelFrame].shape[1]
    )
        

class ClassificationTask(btf.TorchTrainingTask):
    def initialize_task(self, data):
        self.metric_pool = bt.MetricPool().add_sklearn(roc_auc_score)
        self.settings.epoch_count = 10
        self.settings.batch_size = 1000
        self.settings.mini_match_size = None
        self.setup_batcher(data, [get_feature_extractor(), get_binary_label_extractor()])
        self.setup_model(create_factory(100))
        
        
task = ClassificationTask()
result = task.run(get_binary_classification_bundle())
pd.DataFrame(result['output']['history']).set_index('iteration').plot()
```




    <AxesSubplot:xlabel='iteration'>




    
![png](README_images/tg.common.ml.batched_training.factories_output_11_1.png?raw=true)
    


`create_factory` is somewhat of a awkward method that is, essentially, a factory of factories. Moreover, it returns a `lambda` function, which is not compatible with the delivery. Hence, let's consider another way of the factory definition:


```python
class ClassificationNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassificationNetwork, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        X = input['features']
        X = torch.tensor(X.astype(float).values).float()
        X = self.hidden(X)
        X = torch.sigmoid(X)
        X = self.output(X)
        X = torch.sigmoid(X)
        return X
    
    class Factory:
        def __init__(self, hidden_size):
            self.hidden_size = hidden_size
            
        def __call__(self, sample):
            return ClassificationNetwork(
                sample['features'].shape[1], 
                self.hidden_size, 
                sample[btf.Conventions.LabelFrame].shape[1]
            )
        

class ClassificationTask(btf.TorchTrainingTask):
    def initialize_task(self, data):
        self.metric_pool = bt.MetricPool().add_sklearn(roc_auc_score)
        self.settings.epoch_count = 10
        self.settings.batch_size = 1000
        self.settings.mini_match_size = None
        self.setup_batcher(data, [get_feature_extractor(), get_binary_label_extractor()])
        self.setup_model(ClassificationNetwork.Factory(100))
        
        
task = ClassificationTask()
result = task.run(get_binary_classification_bundle())
pd.DataFrame(result['output']['history']).set_index('iteration').plot()
```




    <AxesSubplot:xlabel='iteration'>




    
![png](README_images/tg.common.ml.batched_training.factories_output_13_1.png?raw=true)
    


This way the `ClassificationNetwork.Factory` is a proper factory class, containing the parameters of the to-be-created network. `__call__` method makes the object callable. Placing `Factory` inside `ClassificationNetwork` allows you to import these classes always as a couple, and also allows avoiding excessive naming (`ClassificationNetwork` and `ClassificationNetworkFactory`).

Now, to Optimizator constructor. This is an instance of `CtorAdapter` class that turns a type's constuctor into function with unnamed arguments. It contains: 
* a `type`: either an instance of `type` or a string that encodes type the same way we saw in `tg.common.ml.single_frame`
* `args_names`: mapping from the position of the unnamed parameter to the name of the argument in constructor.
* additional named arguments of the constructor.


```python
task.optimizer_ctor.__dict__
```




    {'type': 'torch.optim:SGD', 'args_names': ('params',), 'kwargs': {'lr': 0.1}}



You can simply change the arguments you need.


```python
class ClassificationTask(btf.TorchTrainingTask):
    def initialize_task(self, data):
        metrics = bt.MetricPool().add_sklearn(roc_auc_score)
        self.metric_pool = bt.MetricPool().add_sklearn(roc_auc_score)
        self.settings.epoch_count = 10
        self.settings.batch_size = 1000
        self.settings.mini_match_size = None
        self.setup_batcher(data, [get_feature_extractor(), get_binary_label_extractor()])
        
        self.optimizer_ctor.type = 'torch.optim:Adam'
        self.optimizer_ctor.kwargs.lr = 0.01
        self.setup_model(ClassificationNetwork.Factory(100))
        
task = ClassificationTask()
result = task.run(get_binary_classification_bundle())
print(task.model_handler.optimizer)
pd.DataFrame(result['output']['history']).set_index('iteration').plot()
```

    Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        eps: 1e-08
        lr: 0.01
        weight_decay: 0
    )





    <AxesSubplot:xlabel='iteration'>




    
![png](README_images/tg.common.ml.batched_training.factories_output_18_2.png?raw=true)
    


## Other ways of defining network

In the end, we need a network factory: a function, that takes a sample batch and creates a network. As we have seen in the example, the batch is actually used in this creation, as it defines the input size of the network. Note that this is, in general, variable, even for the same task and the same dataset: the transformers that perform one-hot encoding on categorical variables are trained on one batch, therefore, different runs may have slightly different amount of columns in categorical features.

The most understandable way of defining the network and the factory was just presented: to create a `torch` component and a separate `Factory` class accepts the parameters in constructor and creates an object in `__call__`, so the `Factory` object is actually callable and can be invoked as a function. 

**This is perfectly normal and functioning way, and we definitely recommend it for the first attempts.** However, it brings an enourmous amount of bad code in the project. If you change your model even slightly, you either need to create a new `Network/Factory` classes (and probably copy-paste the code), or insert flags. The way to prevent this is to find a way to decompose the factory into components and build the more complicated factory from the basic ones. This would be exactly the way `pytorch` itself works, allowing you to assemble network from basic building blocks.

However, the idea of building a `Factory` for each `pytorch` block, which we initially tried, where `Factory` is a descendant of some `AbstractNetworkFactory` class, was a dead end. It brings in way too much of infrastructure code, which should more or less mirror `pytorch` classes. And when you want to add a new component in your network for an experiment, the last thing you want to do is to also write `Factory` for it.

So we have choosen a more subtle way: allow defining factories as just functions, or even as components themselves. We will list several different ways of creating `Factory` out of component, and to combine them together.

For this, let's first create a batch to test our networks:


```python
batch, _ = task.generate_sample_batch_and_temp_data(get_binary_classification_bundle())
batch['features']
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.091268</td>
      <td>-2.069159</td>
      <td>1.270191</td>
      <td>0.990942</td>
      <td>1.614542</td>
      <td>3.324519</td>
      <td>2.653900</td>
      <td>2.550951</td>
      <td>2.275591</td>
      <td>2.254197</td>
      <td>...</td>
      <td>1.878808</td>
      <td>-1.363034</td>
      <td>2.310694</td>
      <td>2.032141</td>
      <td>1.372010</td>
      <td>2.571760</td>
      <td>2.091379</td>
      <td>2.318438</td>
      <td>2.741997</td>
      <td>1.910886</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.825559</td>
      <td>-0.373272</td>
      <td>1.688311</td>
      <td>1.927929</td>
      <td>-0.816927</td>
      <td>-0.477222</td>
      <td>-0.022115</td>
      <td>0.556434</td>
      <td>0.021409</td>
      <td>-0.857298</td>
      <td>...</td>
      <td>1.797949</td>
      <td>-0.383978</td>
      <td>1.537466</td>
      <td>1.919285</td>
      <td>-0.346334</td>
      <td>-0.414532</td>
      <td>-0.143391</td>
      <td>1.105716</td>
      <td>-0.237848</td>
      <td>0.278243</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.575103</td>
      <td>0.425332</td>
      <td>1.568257</td>
      <td>1.573315</td>
      <td>0.978863</td>
      <td>1.075499</td>
      <td>1.364844</td>
      <td>2.053164</td>
      <td>0.975815</td>
      <td>-0.388597</td>
      <td>...</td>
      <td>1.503541</td>
      <td>-0.042597</td>
      <td>1.348655</td>
      <td>1.476818</td>
      <td>0.575486</td>
      <td>1.068639</td>
      <td>0.848784</td>
      <td>1.976310</td>
      <td>1.151458</td>
      <td>0.199561</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.778611</td>
      <td>0.225681</td>
      <td>-0.601828</td>
      <td>-0.781835</td>
      <td>3.355433</td>
      <td>3.444899</td>
      <td>1.917117</td>
      <td>1.464635</td>
      <td>2.936619</td>
      <td>4.898402</td>
      <td>...</td>
      <td>-0.291933</td>
      <td>0.113602</td>
      <td>-0.258640</td>
      <td>-0.567665</td>
      <td>3.502039</td>
      <td>3.823011</td>
      <td>1.972584</td>
      <td>2.197777</td>
      <td>6.021276</td>
      <td>4.866911</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.745868</td>
      <td>-1.160402</td>
      <td>1.779387</td>
      <td>1.844321</td>
      <td>0.307067</td>
      <td>0.557671</td>
      <td>1.372375</td>
      <td>1.441301</td>
      <td>0.010268</td>
      <td>-0.552360</td>
      <td>...</td>
      <td>1.289992</td>
      <td>-1.469313</td>
      <td>1.339664</td>
      <td>1.236776</td>
      <td>0.262246</td>
      <td>-0.299819</td>
      <td>0.609293</td>
      <td>0.746787</td>
      <td>-0.859253</td>
      <td>-0.390551</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>564</th>
      <td>2.107322</td>
      <td>0.686944</td>
      <td>2.065033</td>
      <td>2.369033</td>
      <td>1.079994</td>
      <td>0.234744</td>
      <td>1.948496</td>
      <td>2.338355</td>
      <td>-0.297965</td>
      <td>-0.919415</td>
      <td>...</td>
      <td>1.893321</td>
      <td>0.097499</td>
      <td>1.756248</td>
      <td>2.046472</td>
      <td>0.423341</td>
      <td>-0.260542</td>
      <td>0.660138</td>
      <td>1.649455</td>
      <td>-1.348650</td>
      <td>-0.698174</td>
    </tr>
    <tr>
      <th>565</th>
      <td>1.700331</td>
      <td>2.031720</td>
      <td>1.617935</td>
      <td>1.740531</td>
      <td>0.126477</td>
      <td>-0.004106</td>
      <td>0.694585</td>
      <td>1.275632</td>
      <td>-0.201410</td>
      <td>-1.046472</td>
      <td>...</td>
      <td>1.528421</td>
      <td>2.005692</td>
      <td>1.423580</td>
      <td>1.516228</td>
      <td>-0.668523</td>
      <td>-0.379619</td>
      <td>0.236277</td>
      <td>0.751369</td>
      <td>-0.524403</td>
      <td>-0.959354</td>
    </tr>
    <tr>
      <th>566</th>
      <td>0.695662</td>
      <td>1.992708</td>
      <td>0.669919</td>
      <td>0.578957</td>
      <td>-0.830652</td>
      <td>-0.025125</td>
      <td>0.048300</td>
      <td>0.111796</td>
      <td>-0.803020</td>
      <td>-0.884121</td>
      <td>...</td>
      <td>0.551898</td>
      <td>1.340642</td>
      <td>0.575427</td>
      <td>0.428871</td>
      <td>-0.789344</td>
      <td>0.351056</td>
      <td>0.325611</td>
      <td>0.430624</td>
      <td>-1.094292</td>
      <td>-0.312962</td>
    </tr>
    <tr>
      <th>567</th>
      <td>1.834097</td>
      <td>2.279563</td>
      <td>1.986377</td>
      <td>1.752063</td>
      <td>1.571200</td>
      <td>3.313054</td>
      <td>3.297800</td>
      <td>2.677990</td>
      <td>2.193890</td>
      <td>1.047151</td>
      <td>...</td>
      <td>1.953447</td>
      <td>2.194096</td>
      <td>2.310694</td>
      <td>1.677451</td>
      <td>1.497305</td>
      <td>3.834233</td>
      <td>3.169087</td>
      <td>2.312328</td>
      <td>1.914531</td>
      <td>2.189550</td>
    </tr>
    <tr>
      <th>568</th>
      <td>-1.820280</td>
      <td>1.180334</td>
      <td>-1.829694</td>
      <td>-1.373145</td>
      <td>-3.136430</td>
      <td>-1.146386</td>
      <td>-1.112854</td>
      <td>-1.262821</td>
      <td>-0.814161</td>
      <td>-0.550948</td>
      <td>...</td>
      <td>-1.422709</td>
      <td>0.736784</td>
      <td>-1.448751</td>
      <td>-1.103461</td>
      <td>-1.860624</td>
      <td>-1.176130</td>
      <td>-1.291426</td>
      <td>-1.735169</td>
      <td>-0.043056</td>
      <td>-0.739701</td>
    </tr>
  </tbody>
</table>
<p>455 rows × 30 columns</p>
</div>



Batch contains dataframes, as it's supposed to. But `torch` layers are working with tensors, not with dataframes. So the first component will pick some dataframes from the dictionary, concatenate it and convert to tensors:


```python
features = btf.InputConversionNetwork('features')(batch)
features
```




    tensor([[ 1.0913, -2.0692,  1.2702,  ...,  2.3184,  2.7420,  1.9109],
            [ 1.8256, -0.3733,  1.6883,  ...,  1.1057, -0.2378,  0.2782],
            [ 1.5751,  0.4253,  1.5683,  ...,  1.9763,  1.1515,  0.1996],
            ...,
            [ 0.6957,  1.9927,  0.6699,  ...,  0.4306, -1.0943, -0.3130],
            [ 1.8341,  2.2796,  1.9864,  ...,  2.3123,  1.9145,  2.1896],
            [-1.8203,  1.1803, -1.8297,  ..., -1.7352, -0.0431, -0.7397]])



How to create a factory for this network? Well, unlike `ClassificationNetwork`, this network does not really depend on the input. Thus, it does not require a factory. Some modules in `torch` are also like that, for instance, `Dropout` or `Embedding`.

Then, we can use the class `btf.Perceptron` for one perceptron layer in our network.


```python
btf.Perceptron(features, 10)
```




    Perceptron(
      (linear_layer): Linear(in_features=30, out_features=10, bias=True)
    )



`btf.Perceptron` accepts `input_size` and `output_size`, but both arguments can be `int` or `tensors`. If they are tensors, `btf.Perceptron` will try to deduce the required size out of them. 

Do we need a factory? No: `btf.Perceptron` accepts tensor as a first argument, and other constructor arguments are fixed parameters, so:


```python
from functools import partial

factory = partial(btf.Perceptron, output_size=10)
factory(features)
```




    Perceptron(
      (linear_layer): Linear(in_features=30, out_features=10, bias=True)
    )



But how to combine together all these? We have `FeedForwardNetwork` for this:


```python
factory = btf.FeedForwardNetwork.Factory(
    btf.InputConversionNetwork('features'),
    partial(btf.Perceptron,output_size=10),
    partial(btf.Perceptron, output_size=1)
)
factory(batch)
```




    FeedForwardNetwork(
      (networks): ModuleList(
        (0): InputConversionNetwork()
        (1): Perceptron(
          (linear_layer): Linear(in_features=30, out_features=10, bias=True)
        )
        (2): Perceptron(
          (linear_layer): Linear(in_features=10, out_features=1, bias=True)
        )
      )
    )



Looks good! However, we still had to indicate the output size of the network manually. This logic is implemented in `FullyConnectedNetworkFactory` class:


```python
factory = btf.Factories.FullyConnected([10],'features',btf.Conventions.LabelFrame)
factory(batch)
```




    FeedForwardNetwork(
      (networks): ModuleList(
        (0): FeedForwardNetwork(
          (networks): ModuleList(
            (0): InputConversionNetwork()
            (1): Perceptron(
              (linear_layer): Linear(in_features=30, out_features=10, bias=True)
            )
          )
        )
        (1): Perceptron(
          (linear_layer): Linear(in_features=10, out_features=1, bias=True)
        )
      )
    )



Now, let's try this network with our classification task:


```python
class ClassificationTask(btf.TorchTrainingTask):
    def initialize_task(self, data):
        self.metric_pool = bt.MetricPool().add_sklearn(roc_auc_score)
        self.settings.epoch_count = 10
        self.settings.batch_size = 1000
        self.settings.mini_match_size = None
        self.setup_batcher(data, [get_feature_extractor(), get_binary_label_extractor()])
        self.setup_model(btf.Factories.FullyConnected([10],'features',btf.Conventions.LabelFrame))
        
task = ClassificationTask()
result = task.run(get_binary_classification_bundle())
pd.DataFrame(result['output']['history']).set_index('iteration').plot()
```




    <AxesSubplot:xlabel='iteration'>




    
![png](README_images/tg.common.ml.batched_training.factories_output_34_1.png?raw=true)
    


## Multilabel classification

We will also demonstrate how the system works on multilabel classification. Let's create a bundle from the well-known `iris` dataset.


```python
def get_multilabel_classification_bundle():
    ds = datasets.load_iris()
    features = pd.DataFrame(ds['data'], columns=ds['feature_names'])
    df = pd.DataFrame(ds['target_names'][ds['target']], columns = ['label'])
    df['split'] = bt.train_display_test_split(df, 0.2, 0.2, 'label')
    bundle = bt.DataBundle(index=df, features=features)
    return bundle
    
get_multilabel_classification_bundle().index.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>setosa</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>setosa</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>setosa</td>
      <td>train</td>
    </tr>
    <tr>
      <th>3</th>
      <td>setosa</td>
      <td>test</td>
    </tr>
    <tr>
      <th>4</th>
      <td>setosa</td>
      <td>display</td>
    </tr>
  </tbody>
</table>
</div>




```python
def get_multilabel_extractor():
    label_extractor = (bt.PlainExtractor
                   .build(btf.Conventions.LabelFrame)
                   .index()
                   .apply(take_columns=['label'], transformer=dft.DataFrameTransformerFactory.default_factory())
                  )
    return label_extractor

db = get_multilabel_classification_bundle()
idb = bt.IndexedDataBundle(db.index, db)
test_extractor( get_multilabel_extractor(), idb).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label_setosa</th>
      <th>label_versicolor</th>
      <th>label_virginica</th>
    </tr>
  </thead>
  <tbody>
    <tr>
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
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
class ClassificationTask(btf.TorchTrainingTask):
    def initialize_task(self, data):
        self.metric_pool = bt.MetricPool().add(bt.MulticlassMetrics())
        self.settings.epoch_count = 20
        self.settings.batch_size = 10000
        self.settings.mini_match_size = None
        
        self.setup_batcher(data, [get_feature_extractor(), get_multilabel_extractor()])
        self.optimizer_ctor.kwargs.lr = 1
        self.setup_model(btf.Factories.FullyConnected([],'features',btf.Conventions.LabelFrame))
        
task = ClassificationTask()
result = task.run(get_multilabel_classification_bundle())
pd.DataFrame(result['output']['history']).set_index('iteration').plot()
```




    <AxesSubplot:xlabel='iteration'>




    
![png](README_images/tg.common.ml.batched_training.factories_output_39_1.png?raw=true)
    


## Best practices



### How to get aboard?



### Configuring experiments



Setting the sequence of the experiments is hard. It gets even harder if you don't do it right: in the end, you might end up with lots of jobs at Sagemaker with some metrics, but no one knows what are the parameters of the jobs, or how to use, reuse or reproduce them. To avoid these issues we recommend:

* Configure with parameters, not code. This is not a good practice, when to conduct an experiment, you change something in the code, and then change it back after the experiment. Instead, you put the code, required by the experiment, into the network and isolate it with flags. Ideally, you are able to reproduce any past experiments you had by settings the corresponding parameters.
* If there is a functionality that we may require in other experiments, we isolate it in the separate entities, and the parameters are contained in this entities, _not_ in `TorchTrainingTask` subclass. Examply here is `optimizer_ctor` which contains all the parameters of the optimizer. Don't duplicate settings fields.
* Give meaningful names to the tasks, that represent the important parameters.

To do so, we offer tro-tier initialization. First tier is the Task, such as classification Task. It should be functionable from the start, so use defaults for all the parameters you can:


```python
class ClassificationTask(btf.TorchTrainingTask):
    def __init__(self):
        super(ClassificationTask, self).__init__()
        self.hidden_size = (50,)
    
    def initialize_task(self, data):
        self.metric_pool = bt.MetricPool().add(bt.MulticlassMetrics())
        self.setup_batcher(data, [get_feature_extractor(), get_multilabel_extractor()])
        self.setup_model(btf.Factories.FullyConnected(self.hidden_size,'features',btf.Conventions.LabelFrame))
        
```

Then, we recommend to create the function that accept all the parameters you want to use, and applies them in the corresponding fields:


```python
def create_task(epoch_count=20, network_size=(50,), learning_rate = 1, algorithm ='Adam'):
    task = ClassificationTask()
    task.settings.epoch_count = epoch_count
    task.hidden_size = network_size
    task.optimizer_ctor.kwargs.lr = learning_rate
    task.optimizer_ctor.type='torch.optim:'+algorithm
    return task
    
create_task()
```




    <__main__.ClassificationTask at 0x7f2e402c6b20>



We can safely do this, because all the initialization happens in `late_initialization`, so, after we actually execute the task. Hence, we can modify the parameters after the object has been created.

To maintain the meaningful names, use this class:


```python
from tg.common.delivery.sagemaker import Autonamer

creator = Autonamer(create_task)
tasks = creator.build_tasks(
    network_size=[ (), (10,), (10,10) ],
    learning_rate = [0.1, 0.3, 1],
    algorithm = ['Adam','SGD']
)
tasks[-1].info['name']
```




    'NS10-10-LR1-ASGD'



`Autonamer` accepts the ranges of each argument of the `create_task`, then runs to all possible combinations, create a task with these parameters and also assigns the name to it, trying to abbreviate the name of the argument and shorten it's value.


```python
results = {}
for task in Query.en(tasks).feed(fluq.with_progress_bar()):
    result = task.run(get_multilabel_classification_bundle())
    results[task.info['name']] = pd.DataFrame(result['output']['history']).set_index('iteration').accuracy_test
```


      0%|          | 0/18 [00:00<?, ?it/s]



```python
from matplotlib import pyplot as plt
_, ax = plt.subplots(1,1,figsize=(20,10))
rdf = pd.DataFrame(results)
rdf.plot(ax=ax)
```




    <AxesSubplot:xlabel='iteration'>




    
![png](README_images/tg.common.ml.batched_training.factories_output_50_1.png?raw=true)
    



```python
rdf.iloc[-1].sort_values(ascending=False).plot(kind='barh')
```




    <AxesSubplot:>




    
![png](README_images/tg.common.ml.batched_training.factories_output_51_1.png?raw=true)
    


Now note, that all the parameters are available as fields somewhere deep in the `task` object. That means that hyperparameter optimization, explained in `tg.common.ml.single_frame` remains available.



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




```python
from tg.common.ml import batched_training as bt

plain_bundle = bt.DataBundle(index=df)
```

First, let's define the extractors:


```python
from tg.common.ml.batched_training import factories as btf
from tg.common.ml import dft

label_extractor = bt.PlainExtractor.build('label').index().apply(take_columns=btf.Conventions.LabelFrame)
features_extractor = bt.PlainExtractor.build('features').index().apply(
    take_columns=[f for f in df.columns if f.startswith('word')],
    transformer=dft.DataFrameTransformerFactory.default_factory()
)
plain_batch = bt.Batcher.generate_sample(plain_bundle, [label_extractor, features_extractor])
plain_batch['features'].head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word_0_A</th>
      <th>word_1_A</th>
      <th>word_1_B</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
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
      <td>1.0</td>
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
      <td>1.0</td>
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
      <td>1.0</td>
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




```python
plain_batch['label'].head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
    </tr>
    <tr>
      <th>sentence_id</th>
      <th></th>
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
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.metrics import roc_auc_score
from tg.common import Logger

Logger.disable()

class PlainTask(btf.TorchTrainingTask):
    def initialize_task(self, data):
        metrics = bt.MetricPool().add_sklearn(roc_auc_score)
        self.settings.batch_size = 1000
        self.settings.mini_match_size = None
        self.metric_pool = metrics
        
        label_extractor = bt.PlainExtractor.build('label').index().apply(take_columns=btf.Conventions.LabelFrame)
        features_extractor = bt.PlainExtractor.build('features').index().apply(
            take_columns=[f for f in df.columns if f.startswith('word')],
            transformer=dft.DataFrameTransformerFactory.default_factory()
        )
        self.setup_batcher(data, [features_extractor, label_extractor])
        
        self.optimizer_ctor.kwargs.lr = 1
        self.setup_model(btf.Factories.FullyConnected([100],'features',btf.Conventions.LabelFrame))
        
def run(task, bundle, epoch_count = 100, name = 'roc_auc'):
    task.settings.epoch_count = epoch_count
    result = task.run(bundle)
    df = pd.DataFrame(result['output']['history']).set_index('iteration')
    series = df.roc_auc_score_display
    series.name = str(name)
    return series

run(PlainTask(), bt.DataBundle(index=df)).plot()
```




    <AxesSubplot:xlabel='iteration'>




    
![png](README_images/tg.common.ml.batched_training.context_output_10_1.png?raw=true)
    


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
          .select(lambda z: run(PlainTask(), bt.DataBundle(index=df), epoch_count=500, name=str(z)))
          .to_list()
         )
pd.DataFrame(curves).transpose().plot()
pass
```


      0%|          | 0/5 [00:00<?, ?it/s]



    
![png](README_images/tg.common.ml.batched_training.context_output_13_1.png?raw=true)
    


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
        bundle = bt.DataBundle(index=df)
        roc = run(PlainTask(), bundle, name=i)
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
class SequentialTask(btf.TorchTrainingTask):
    def __init__(self, context_extractor, network_factory, lr = 1):
        super(SequentialTask, self).__init__()
        self.context_extractor = context_extractor
        self.network_factory = network_factory
        self.lr = lr
    
    def initialize_task(self, data):
        metrics = bt.MetricPool().add_sklearn(roc_auc_score)
        self.settings.batch_size = 1000
        self.settings.mini_match_size = None
        self.metric_pool = metrics
        
        label_extractor = bt.PlainExtractor.build('label').index().apply(take_columns=btf.Conventions.LabelFrame)
        self.setup_batcher(data, [self.context_extractor, label_extractor])
        
        self.optimizer_ctor.kwargs.lr = self.lr
        self.setup_model(self.network_factory)
        

    
task = SequentialTask(
    build_context_extractor(
        db.additional_information.context_length,
        btc.PivotAggregator()
    ),
    btf.Factories.FullyConnected([100],'features',btf.Conventions.LabelFrame)
)

run(task, db).plot()
pass
```


    
![png](README_images/tg.common.ml.batched_training.context_output_33_0.png?raw=true)
    


So, the system achieved the same performance at the same time, as a naive implementation, confirming the correctness of the implementation (of course, the used classes are also covered by tests).

Let's now explore the task a bit further and see how the context length affects the performance:


```python
curves = []
for i in Query.en(range(1,db.additional_information.context_length+1)).feed(fluq.with_progress_bar()):
    task = SequentialTask(
        build_context_extractor(i, btc.PivotAggregator()),
        btf.Factories.FullyConnected([100],'features',btf.Conventions.LabelFrame)
    )
    curves.append(get_roc_auc_curve(i, task, db))
pd.DataFrame(curves).transpose().plot()
pass
```


      0%|          | 0/4 [00:00<?, ?it/s]



    
![png](README_images/tg.common.ml.batched_training.context_output_35_1.png?raw=true)
    


Unsurprisingly, we see that if the context is smaller that the actual length of the sentences in our langauge $L$, the performance decreases.

`PivotAggregator` is the most memory-consuming way of representing the contextual data. In this example it's fine, but if context consists of dozens of samples, each having dozens of extracted columns, `PivotAggregator` will produce a very huge matrix that may overfill the memory. 

This is why you may also want to use other aggregators. For instance, `GroupByAggregator` will process the `features` dataframe with grouping by `sample_id` and applying the aggregating functions:


```python
context_extractor = build_context_extractor(i, btc.GroupByAggregator(['mean','max']))
task = SequentialTask(
    context_extractor,
    btf.Factories.FullyConnected([100],'features',btf.Conventions.LabelFrame)
)
_ = task.generate_sample_batch_and_temp_data(db)
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
    task = SequentialTask(
        build_context_extractor(i, btc.GroupByAggregator(['mean', 'max'])),
        btf.Factories.FullyConnected([100],'features',btf.Conventions.LabelFrame)
    )
    curves.append(get_roc_auc_curve(i, task, db))
pd.DataFrame(curves).transpose().plot()
pass
```


      0%|          | 0/4 [00:00<?, ?it/s]



    
![png](README_images/tg.common.ml.batched_training.context_output_40_1.png?raw=true)
    


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
task = SequentialTask(
    context_extractor,
    btf.Factories.FullyConnected([100],'features',btf.Conventions.LabelFrame)
)
_ = task.generate_sample_batch_and_temp_data(db)
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
    task = SequentialTask(
        build_context_extractor(i, CustomAggregator()),
        btf.Factories.FullyConnected([100],'features',btf.Conventions.LabelFrame)
    )
    curves.append(get_roc_auc_curve(i, task, db))
pd.DataFrame(curves).transpose().plot()
pass
```


      0%|          | 0/4 [00:00<?, ?it/s]



    
![png](README_images/tg.common.ml.batched_training.context_output_43_1.png?raw=true)
    


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
        finalizer = btc.LSTMFinalizer()
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

First, let's create a batch:


```python
batch = bt.Batcher.generate_sample(db, [label_extractor, lstm_extractor])
{key: type(value) for key, value in batch.bundle.data_frames.items()}
```




    {'label': pandas.core.frame.DataFrame,
     'features': tg.common.ml.batched_training.factories.networks.basics.AnnotatedTensor}



Then create a factory, and also network to see how it looks like:


```python
from functools import partial

lstm_factory = btf.Factories.Tailing(
    btf.FeedForwardNetwork.Factory(
        btf.InputConversionNetwork('features'),
        partial(btc.LSTMNetwork, hidden_size=10),
    ),
    btf.Conventions.LabelFrame
)

network = lstm_factory(batch)
network
```




    FeedForwardNetwork(
      (networks): ModuleList(
        (0): FeedForwardNetwork(
          (networks): ModuleList(
            (0): InputConversionNetwork()
            (1): LSTMNetwork(
              (lstm): LSTM(3, 10)
            )
          )
        )
        (1): Perceptron(
          (linear_layer): Linear(in_features=10, out_features=1, bias=True)
        )
      )
    )




```python
task = SequentialTask(
    build_lstm_extractor(db.additional_information.context_length),
    lstm_factory,
)

run(task, db).plot()
pass
```


    
![png](README_images/tg.common.ml.batched_training.context_output_60_0.png?raw=true)
    


We see that after 100 iterations the quality is low, but growing, while previous systems have stabilized at this point. So, we will train this network for a longer time.


```python
run(task, db, epoch_count=1000).plot()
pass
```


    
![png](README_images/tg.common.ml.batched_training.context_output_62_0.png?raw=true)
    


In nearly all the cases we run this cell, the quality of the "plain" networks was surpassed, often reaching the 0.9 value without stabilization. It might be the sign that this network architecture is indeed superior in comparison with the other, but this is anyway not the point we're trying to make. The point is that the classes, used for the data transformation, are working correctly and are ready to be used in conjustion with LSTM and other recurrent neural networks.

## Assembly Point

We see that there is a lot of ways to configure networks and extractors for contextual processing. Also, there are actually multiple ways of processing 3-dimentional tensors: LSTM, LSTM with attention, attention only, or Dropout. To easy coordinate the networks and extractors, as well as to create richer networks, `AssemblyPoint` can be used.

`AssemblyPoint` is a factory that produces both extractors and network factories in a coordinated way. Most of the settings will be filled with some default values that allow `AssemblyPoint` work _somehow_ (but not necesserily the way that is optimal for your task).




```python
ibundle.bundle.additional_information.context_length
```




    4




```python
ap = btc.ContextualAssemblyPoint(
    name = 'features',
    context_builder = SentenceContextBuilder(),
    extractor = bt.PlainExtractor.build('word').index().apply(
        take_columns=['word'], 
        transformer = dft.DataFrameTransformerFactory.default_factory())
)
ap.context_length = ibundle.bundle.additional_information.context_length

```

Let's write the method to ensure the extractor works with the network:


```python
def test_assembly_point(ap):
    extractor = ap.create_extractor()
    network_factory = ap.create_network_factory()
    batch = bt.Batcher.generate_sample(db, [extractor])
    network = network_factory(batch)
    result = network(batch)
    response = {
        'features_type': type(batch['features']),
        'features_shape': batch['features'].shape,
        'network_output_shape': result.shape,
        'network': network
    }
    return response
    
test_assembly_point(ap)
```




    {'features_type': pandas.core.frame.DataFrame,
     'features_shape': (10, 17),
     'network_output_shape': torch.Size([10, 20]),
     'network': FeedForwardNetwork(
       (networks): ModuleList(
         (0): InputConversionNetwork()
         (1): Perceptron(
           (linear_layer): Linear(in_features=17, out_features=20, bias=True)
         )
       )
     )}




```python
ap.reduction_type = btc.ReductionType.Dim3
test_assembly_point(ap)
```




    {'features_type': tg.common.ml.batched_training.factories.networks.basics.AnnotatedTensor,
     'features_shape': (4, 10, 3),
     'network_output_shape': torch.Size([10, 20]),
     'network': FeedForwardNetwork(
       (networks): ModuleList(
         (0): InputConversionNetwork()
         (1): LSTMNetwork(
           (lstm): LSTM(3, 20)
         )
       )
     )}




```python
ap.dim_3_network_factory.droupout_rate = 0.5
test_assembly_point(ap)
```




    {'features_type': tg.common.ml.batched_training.factories.networks.basics.AnnotatedTensor,
     'features_shape': (4, 10, 3),
     'network_output_shape': torch.Size([10, 20]),
     'network': FeedForwardNetwork(
       (networks): ModuleList(
         (0): InputConversionNetwork()
         (1): Dropout3d(p=0.5, inplace=False)
         (2): LSTMNetwork(
           (lstm): LSTM(3, 20)
         )
       )
     )}



The use case for the `AssemblyPoint` is:
* create a field of `AssemblyPoint` type in your subclass of `TorchTrainingTask`, e.g., `assebly_point`
* externally adjust fields of `assembly_point` to configure it
* in `initialize_task`, use assemply point to generate extractors and network factories. Augment the network factory with a tailing that ensures the right dimentionality of the output, manually or with `btf.Factories.Tailing` 
* you can even use multiple assemble points! Each will create extractor and head network for the part of the input data, and then you'll assemble these heads into the full network.



# 4.1. Packages and Containers (tg.common.delivery.delivery)

## Overview

In this part, we will deliver the featurization job to a remote server and execute it there. This actually can be done with just few lines of code. But we will show a lot of the process "under the hood" to make you familiar with it, and to explain why do we have this setup.

Delivery is the most fundamental purpose of Training Grounds. It is extremely easy to write _some_ data science code, that is executable on your local machine. It is not so easy though to then deliver this code to a remote server (be it server for training or a web-server that exposes model to the world) so that everything continues to work.

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

We may want different versions of a model to be able to run at the same time. But how can we do that, if the models are represented as packages? In Python, we cannot have two modules with the same name installed at the same time. Thus, they have to have different name. This is why Training Grounds itself is not a Python package, but a folder inside your project. 

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

The name of the TG is actually `UID.tg`, with different UID in each package. Hence, several versions of TG can be used at the same time! But that brings another limitation that must be observed inside `tg` folder: all the references inside TG must be relative. They cannot refer to `tg`, because `tg` will become `UID.tg` in the runtime on the remote server.


### Hot Module Replacement

Now, the question arises, how to use this package. Sometimes we want UID to be created dynamically, and in this case we cannot write something like this:

```
from UID.tg import *
```

The solution is to install the module during runtime. During this process, the name becomes known, and then we can dynamically import from the module. Of course, importing classes or methods would not be handy, but remember that deliverables are objects, and these objects are pickled as the module resources. So all we need to do is to unpickle these objects, and all the classes and methods will be loaded dynamically by unpickler. This work is done by `EntryPoint` class.

However, with Packaging you can also create packages with predictable names, publish them with PyPi and export locally.

#### Note for advanced users

When package is created, we pickle the objects under the local version of TG, thus, the classes are unavoidably pickled as `tg.SomeClass`, but we want to unpickle them as `UID.tg.SomeClass`. How is this achived? Fortunately, pickling allows you to do some manipulations while pickling/unpickling, and so we just replace all `tg.` prefixes to `UID.tg.` while building a package (UID is already known at this time).

It is also possible to do same trick when unpickling: if you want to transfer the previously packaged object into the current `tg` version, this is possible. Of course, it's on your responsibility to ensure that current TG is compatible with an older version. Later we will discuss a use case for that.

## Packaging



Packaging allows you to create a Python package with the source code and pickled job (arbitrary class with `run` method). For this package to work correctly, the job needs to be defined withing `tg` folder, so we will use an example job from training grounds. This job simply outputs `SUCCESS` with the Logger.


```python
from tg.common.delivery.delivery.example_job import ExampleJob

job = ExampleJob()
job.run()
```

    2022-12-28 14:24:44.951052 INFO: SUCCESS


`Packaging` class is a representation of all the settings, required for packaging. In the constructor, it only accepts the absolutely necessary values:


```python
from tg.common.delivery.delivery import Packaging

packaging = Packaging(
    name = 'example_job',
    version = '0.0.0',
    payload = dict(job = job)
)
packaging.silent = True
```

There are many fields in `Packaging` class, but most of them you don't need to adjust. The `dependencies` field is, however, important: it shows the python dependencies the package will have:


```python
packaging.dependencies
```




    ({'min': ['boto3', 'yo_fluq_ds', 'simplejson']},)



By default, it contains only the dependencies required by `tg.common` itself; not all the dependencies required, e.g., by `tg.common.ml`. You have to manually add all the dependencies you are using, preferrably with the versions. 

Generally, all the packaging code is "semi-finished" products: when used in your projects, it is recommended to create a class that takes care of all delivery-related processes in one method, making necessary calls and adjusting settings. The strategies for this will be discussed in the next demos.

Now, let's create a package file:


```python
packaging.make_package()
pass
```

    warning: no files found matching '*.yml' under directory 'example_job__0_0_0'
    warning: no files found matching '*.rst' under directory 'example_job__0_0_0'
    warning: sdist: standard file not found: should have one of README, README.rst, README.txt, README.md
    


`make_package` stores the file in the local system, and now we will install it "on the fly". As a result, we will get `EntryPoint` object:


```python
from tg.common.delivery.delivery import install_package_and_get_loader

entry_point = install_package_and_get_loader(packaging.package_location, silent = True)
{k:v for k,v in entry_point.__dict__.items() if k!='resources_location'}
```

    WARNING: Skipping example-job as it is not installed.





    {'name': 'example_job',
     'version': '0.0.0',
     'module_name': 'example_job__0_0_0',
     'tg_import_path': 'example_job__0_0_0.tg',
     'original_tg_import_path': 'tg'}



Now we will load the job from the package. 


```python
loaded_job = entry_point.load_resource('job')
print(type(job))
print(type(loaded_job))
```

    <class 'tg.common.delivery.delivery.example_job.ExampleJob'>
    <class 'example_job__0_0_0.tg.common.delivery.delivery.example_job.ExampleJob'>


Note that:
    
  * the classes of `job` and `loaded_job` are different in located in the different models.
  * the created module is not `example_job`, but `example_job__0_0_0`. The reason for this is that we may want to package and run different versions of `ExampleJobs` within one process, and we don't want the module with `0.0.1` version to remove the module with `0.0.1` version. If you don't want this behaviour, adjust `packaging.human_readable_module_name` parameter.

## Containering

Although we could just run the package at the remote server via ssh, the more suitable way is to use Docker. Training Grounds defines methods to build the docker container out of the package.

Most of the container's settings can be inherited from the package, so it's the easiest way to create a containering object:


```python
from tg.common.delivery.delivery import Containering


containering = Containering.from_packaging(packaging)
containering.silent = True
```

A comment regarding dependencies: when building the container, we first pre-install the dependencies, specified in the `Containering` object, then copy package and install the package (and this triggers installation of dependencies, specified in the `Packaging` object). This allows us to reuse the same Docker layer for many containering process, so, if your dependencies are stable and you build many containers with different jobs, it will save a great deal of time.

Let's build the container:


```python
containering.make_container(packaging)
```

    sha256:eec5c21e38845c5a7881c373f5725f701b0bc3b3e1c28106120b1897d862a44c





    <tg.common.delivery.delivery.containering.Containering at 0x7f7c141e1430>



Now we can run the container locally:


```python
containering.image_name, containering.image_tag
```




    ('example_job', '0.0.0')




```python
!docker run example_job:0.0.0
```

    2022-12-28 13:25:06.816628 INFO: Welcome to Training Grounds!
    2022-12-28 13:25:06.816758 INFO: Loading job
    2022-12-28 13:25:06.817456 INFO: Job of type <class 'example_job__0_0_0.tg.common.delivery.delivery.example_job.ExampleJob'> is loaded
    2022-12-28 13:25:06.817570 INFO: Job has `run` attribute
    2022-12-28 13:25:06.817628 INFO: SUCCESS
    2022-12-28 13:25:06.817675 INFO: Job has exited successfully
    2022-12-28 13:25:06.817719 INFO: DONE. Exiting Training Grounds.




# 4.2. Delivery via SSH (tg.common.delivery.ssh_docker)

One of the scenarios for delivery is to start the job at a remote docker server via SSH. Training Grounds contain several classes that facilitate this process.

First, we have `SSHDockerOptions`. This class contains the settings on how to run the job in the docker container: which environmental variables are to propagate from the local machine to the remote one, memory and CPU limits. 

Second, we have `SSHDockerConfig`: a comprehensive configuration for the procedure that contains `Packaging`, `Containering` and `SSHDockerOptions`, as well as the address of the remote host and the username. 

This `SSHDockerConfig` is an argument for `Executors`, which actually execute the job. We have:
  
  * `AttachedExecutor` that runs the job in the same process, without any docker at all.
  * `LocalExecutor` that runs the job in the local docker
  * `RemoteExecutor` that runs the job in the remote docker, using SSH.
  
These three executors help to debug the job. First, we can run `AttachedExecutor` to make sure that the job itself works. By running it in `LocalExecutor` we make sure that packaging and containering work, e.g.:

* Your job is serializable. This is usually achievable by not using lambda syntax.
* All the code the job uses is located inside the TG folder, and if all the references are relative. If something is wrong, you will see the import error.
* If the environmental variables are carried to docker correctly.
* If you have sufficient permissions to start docker

Finally, `RemoteExecutor` will peform the same functionatily remotely.  The only problems you should have at these stage are permissions:

  * to push to your docker registry
  * to connect to the remote machine via SSH
  * to execute docker run at the remote machine
  
The best way to actually use this code in your project is to write a class, e.g., `SSHDockerRoutine`, in the following way:


```python
from tg.common.delivery.ssh_docker import (SSHAttachedExecutor, SSHLocalExecutor, SSHRemoteExecutor, 
                                           SSHDockerOptions, SSHDockerConfig)
from tg.common.delivery.delivery import Packaging, Containering

variable_name = 'EXAMPLE_VARIABLE'

class SSHDockerRoutine:
    def __init__(self, job):
        self.job = job
        name = type(job).__name__.lower()
        packaging = Packaging(name, '0.0.0', dict(job=job))
        packaging.silent = True
        containering = Containering.from_packaging(packaging)
        containering.silent = True
        options = SSHDockerOptions([variable_name])
        self.config =  SSHDockerConfig(packaging, containering, options, None, None)

    def attached(self):
        return SSHAttachedExecutor(self.config)

    def local(self):
        return SSHLocalExecutor(self.config)
    
        
```

In the `_create_config` method you can place all the logic regarding dependencies, secrets, etc. After this, you can simply use `SSH3DockerRoutine` to run the job remotely. 

Let's run the `ExampleJob` with attached executor:


```python
from tg.common.delivery.delivery.example_job import ExampleJob
import os

os.environ[variable_name] = 'TEST'

job = ExampleJob([variable_name])
routine = SSHDockerRoutine(job)
routine.attached().execute()
```

    2022-12-28 14:25:11.493742 INFO: Variable EXAMPLE_VARIABLE is found: True
    2022-12-28 14:25:11.494948 INFO: SUCCESS


And now, with local:


```python
routine.local().execute()
```

    warning: no files found matching '*.yml' under directory 'examplejob__0_0_0'
    warning: no files found matching '*.rst' under directory 'examplejob__0_0_0'
    warning: sdist: standard file not found: should have one of README, README.rst, README.txt, README.md
    


    sha256:6a9d1deca68272221c3ac4154fe538a0504f2ce9c5c47940dce82b3a8c71024b
    2022-12-28 13:25:22.692513 INFO: Welcome to Training Grounds!
    2022-12-28 13:25:22.692604 INFO: Loading job
    2022-12-28 13:25:22.693027 INFO: Job of type <class 'examplejob__0_0_0.tg.common.delivery.delivery.example_job.ExampleJob'> is loaded
    2022-12-28 13:25:22.693102 INFO: Job has `run` attribute
    2022-12-28 13:25:22.693194 INFO: Variable EXAMPLE_VARIABLE is found: True
    2022-12-28 13:25:22.693262 INFO: SUCCESS
    2022-12-28 13:25:22.693328 INFO: Job has exited successfully
    2022-12-28 13:25:22.693394 INFO: DONE. Exiting Training Grounds.


As we can see, the environment variable was sucessfully transferred from the notebook environment to the docker's environment.



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




    <matplotlib.legend.Legend at 0x7f6336de52b0>




    
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
      <td>50.930936</td>
      <td>24.669352</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fare</td>
      <td>39.013063</td>
      <td>26.826770</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fare</td>
      <td>40.308628</td>
      <td>23.785594</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fare</td>
      <td>49.674604</td>
      <td>24.626052</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fare</td>
      <td>44.814162</td>
      <td>29.408505</td>
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
      <td>38.011881</td>
      <td>50.822352</td>
      <td>44.417117</td>
      <td>6.405235</td>
      <td>22.013058</td>
      <td>29.000021</td>
      <td>25.506539</td>
      <td>3.493481</td>
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
