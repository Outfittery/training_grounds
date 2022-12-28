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
