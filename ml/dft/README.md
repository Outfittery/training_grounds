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

    2022-08-09 09:26:12.708615+00:00 WARNING: Missing column in ContinuousTransformer
    2022-08-09 09:26:12.716127+00:00 WARNING: Missing column in ContinuousTransformer


    Traceback (most recent call last):
      File "/tmp/ipykernel_10529/2155061853.py", line 14, in <module>
        tr.transform(test_df)
      File "/home/yura/Desktop/repos/lesvik-ml/tg/common/ml/dft/architecture.py", line 48, in transform
        for res in transformer.transform(df):
      File "/home/yura/Desktop/repos/lesvik-ml/tg/common/ml/dft/column_transformers.py", line 90, in transform
        missing = self.missing_indicator.transform(subdf)
      File "/home/yura/anaconda3/envs/lesvik/lib/python3.8/site-packages/sklearn/impute/_base.py", line 885, in transform
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

    2022-08-09 09:26:12.796480+00:00 WARNING: Missing column in ContinuousTransformer
    2022-08-09 09:26:12.812927+00:00 WARNING: Missing column in ContinuousTransformer
    2022-08-09 09:26:12.873563+00:00 WARNING: Unexpected None in MissingIndicatorWithReporting
    2022-08-09 09:26:12.881915+00:00 WARNING: Unexpected None in MissingIndicatorWithReporting
    2022-08-09 09:26:12.883404+00:00 WARNING: Unexpected None in MissingIndicatorWithReporting





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

    {"@timestamp": "2022-08-09 09:26:12.966395+00:00", "message": "Missing column in ContinuousTransformer", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/lesvik-ml/tg/common/ml/dft/column_transformers.py", "path_line": 75, "column": "Pclass"}
    {"@timestamp": "2022-08-09 09:26:12.977181+00:00", "message": "Missing column in ContinuousTransformer", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/lesvik-ml/tg/common/ml/dft/column_transformers.py", "path_line": 75, "column": "Parch"}
    {"@timestamp": "2022-08-09 09:26:13.083818+00:00", "message": "Unexpected None in MissingIndicatorWithReporting", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/lesvik-ml/tg/common/ml/dft/miscellaneous.py", "path_line": 36, "column": "Pclass"}
    {"@timestamp": "2022-08-09 09:26:13.084559+00:00", "message": "Unexpected None in MissingIndicatorWithReporting", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/lesvik-ml/tg/common/ml/dft/miscellaneous.py", "path_line": 36, "column": "Parch"}
    {"@timestamp": "2022-08-09 09:26:13.088141+00:00", "message": "Unexpected None in MissingIndicatorWithReporting", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/lesvik-ml/tg/common/ml/dft/miscellaneous.py", "path_line": 36, "column": "Fare"}





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

    {"@timestamp": "2022-08-09 09:26:13.469973+00:00", "message": "Unexpected value in MostPopularStrategy", "levelname": "WARNING", "logger": "tg", "path": "/home/yura/Desktop/repos/lesvik-ml/tg/common/ml/dft/column_transformers.py", "path_line": 122, "column": "Embarked", "value": "NONE"}





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
