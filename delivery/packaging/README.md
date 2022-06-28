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
