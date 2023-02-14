# 3. Machine Learning (tg.common.ml)

Machine learning is supported by the following modules:

* `tg.common.ml.dft`: a handy decorator over `sklearn` that applies transformers to the dataframes and keeps the names of the columns. Greatly improves debugging of the machine learning pipelines and simplifies data cleaning.
* `tg.common.ml.single_frame`: pipeline that applies simpler ML-algorithms (logistic regression, XGBoost) to the data. The requirement is that the data must fit to the memory after transformation all at once.
* `tg.common.ml.batched_training`: pipeline that applies more complex algorithms, typically neural networks, to the data that do not fit in the memory all at once after transformation. 

