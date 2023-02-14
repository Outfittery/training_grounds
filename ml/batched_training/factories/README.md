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
