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
