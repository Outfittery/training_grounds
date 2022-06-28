# 2.3. Featurization Jobs and Datasets (tg.common.datasets.featurization)

In the `tg.common.datasets.access` and `tg.common.datasets.selectors`, we covered `DataSource` and `Selector` classes to build a tidy dataset from the external data source. For small datasets that can be build in several minutes, these two components are fine and you don't need anything more. However, for the bigger datasets, additional questions arise, like:

* Sometimes the data set is too large to hold in memory. Since selector produces rows one by one, it's not a big problem: we can just separate them into several smaller dataframes
* Sometimes we want to exclude some records from the dataset, or produce several rows per one object
* Sometimes we actually do not want the resulting dataframe, but some aggregated statistics instead
* And finally, sometimes we do not want to execute this procedure on our local machine. Instead, we want to deliver it to the cloud.

`tg.common.datasets.featurization` addresses these questions, offering `FeaturizationJob` and `UpdatableFeaturizationJob`, which are production-ready classes to create datasets on scale: 

* `FeaturizationJob` creates the whole dataset and cannot update its rows.
* `UpdatableFeaturizationJob` creates the dataset or updates it, if only some rows are changed.
