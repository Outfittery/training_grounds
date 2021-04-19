# Training Grounds

## Overview

Training Grounds (TG) is a data science framework, developed in Outfittery Gmbh (www.outfittery.de).
Currently, it is used in several projects for data preprocessing pipelines and machine learning pipelines design.
The main tools available are:

1. Featurization: easy-to-write pure functions that convert the unstructured data into tidy dataset.
1. Data Frame Transformers: a slim wrapper around `sklearn` that applies the transformations to `pandas.Dataframe` 
and keeps the column names.
1. Training: SOLID-based architecture to define machine-learning pipelines, including test/train splitting, 
training itself, evaluation and hyperparameter tuning.
1. Delivery: the way to package the code, send it to the remote server and execute there. Here, TG provides a flexible way of delivery
(no integration with the version control system or any other external software is required), 
reliable versioning technique and hot-module replacement.

Our main principles were:

1. Use the existing solution to the maximum extent and write the lightest wrappers possible.
1. SOLID to provide testability, extendability and reusability of the solutions.
1. "Open framework": we do not claim that we offer all components you will ever need; 
rather, we offer you the architecture where you can build components for your use cases, and the components we used for ours.
1. Minimizing complexity gap: instead of domain specific languages or visual programming that are often easy in the training phase,
but are hard to adopt for the real use cases, not covered by the manual, we offer a code-first approach that behaves 
the same way on all the stages of adoption.

This repository contains:

1. The source code of the framework
1. The comprehensive set of Demos that covers the platform functionality. Demos are Jupyter Notebooks, so you can not only run them,
but also play around and explore the functionality yourself.
1. Tests that cover substantial part of the code, especially in the crucial parts, but not 100% of it.

This repository **is not** a Python module. This is due to technical requirements of the delivery subsystem, which
is covered in the Demo in full details. Two installation options are available.

## Fast installation

* run `pip install training-grounds` in your environment
* checkout or download this repository on you machine
* run `jupyter notebook` in terminal, open the Demos in `tg/common/demos` folder.

On your local machine, you will be able to fully use TG without any limitations.
As for delivery, you will only be able to deliver the objects that are composed entirely from build-in TG classes.
This limitation is not something you should necessarily be concerned at your first steps.

## Full-fledged installation

* Create a git repository for your project. We will refer to the folder of the repository as `/`
  * If you don't want to use `git`, just create `/` folder.
* Add Training Grounds as a submodule to `/tg/common` folder.
  * If you don't want to use `git`, just checkout TG repository to `/tg/common` folder
  * You can choose another name for the folder instead of `/tg/common`, but the tests and demos won't work in this case
* create an empty `tg/__init__.py`. 
  * Otherwise `tg` will not be recognized as module, so you wont be able to import from `tg.common`
* create `/setup.py`
  * You may use the following template:
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
  * add to the `setup.py` the requirements for TG that are listed in `tg/common/requirements.txt`. 
    You don't have to install all of them. E.g. if you don't use machine learning in the project, you don't have to install torch or sklearn.
* create and activate the virtual environment 
  * E.g., in Anaconda: 
    * `conda create --name tg python=3.8`
    * `conda activate tg`
* Go to `/` directory in Terminal and execute `pip install -e .` command
* Execute `pip install -r tg/common/requirements.txt` command
* To start the demos, type `jupyter notebook` in terminal, open the Demos in `/tg/common/demos` folder.
 
### Tests

Tests are located in `tg/common/test_common`. Some of them use external dependencies:
* Docker must be installed and accessible without `sudo`
 
### Folders

There are two folders created by TG:

* `/temp`. It contains files that can be deleted at any time.
* `/data-cache`. It contains downloaded data. Deleting it will cause re-downloading.

These folders should be added to `.gitignore`. To access them from within code, use `tg.common.Loc` object.

