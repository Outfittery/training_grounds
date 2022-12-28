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

