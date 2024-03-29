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
