# 2.2. Selectors (tg.common.datasets.selectors)

## Overview

Selectors are objects that:
* define a pure function that transforms a data object into a row in the dataset
* track errors and warnings that happen during this conversion
* fully maintain the inner structure of selectors, making it possible to e.g. visualize the selector

**Note**: selectors are slow! They are not really aligned for the processing of hundreds of gygabytes of data. If this use case arises:
* they potentially can be parallelized in a PySpark 
* they potentially can be partially translated into e.g. PrestoSQL queries, since they maintain the inner structure


## Basic Selectors

We will use the distorted Titanic dataset from the previous demo, and apply various selectors to one of the Data Objects.


```python
from tg.common.datasets.access import ZippedFileDataSource, CacheableDataSource, CacheMode

source = CacheableDataSource(
    inner_data_source = None,
    file_data_source = ZippedFileDataSource(path='./titanic.zip'),
    default_mode=CacheMode.Use
)
obj = source.get_data().skip(11).first()
obj
```




    {'id': 12,
     'ticket': {'ticket.id': '113783', 'fare': 26.55, 'Pclass': 1},
     'passenger': {'Name': 'Bonnell, Miss. Elizabeth',
      'Sex': 'female',
      'Age': 58.0},
     'trip': {'Survived': 1,
      'SibSp': 0,
      'Patch': 0,
      'Cabin': 'C103',
      'Embarked': 'S'}}



Let's start with simply selecting one field:


```python
from tg.common.datasets.selectors import Selector

selector = (Selector()
            .select('id')
            )

selector(obj)
```




    {'id': 12}



`Selector` class is a high-level abstraction, that allows you defining the featurization function with a `Fluent API`-interface. `Selector` is building a complex object of interconnected smaller processors, and we will look at these processors a little later. We may consider `Selector` on a pure syntax level: how exactly this or that use case can be covered with it. 

We can rename the field as follow:


```python
selector = (Selector()
            .select(passenger_id = 'id')
            )

selector(obj)
```




    {'passenger_id': 12}



We can select nested fields several syntax options:


```python
from tg.common.datasets.selectors import Selector, FieldGetter, Pipeline

selector = (Selector()
            .select(
                'passenger.Name',
                ['passenger','Age'],
                ticket_id = ['ticket',FieldGetter('ticket.id')],
                sex = Pipeline(FieldGetter('passenger'), FieldGetter('Sex'))
            ))
selector(obj)
```




    {'ticket_id': '113783',
     'sex': 'female',
     'Name': 'Bonnell, Miss. Elizabeth',
     'Age': 58.0}



* the first one (for `Name`) represents the highest level of abstraction, it is very easy to define lots of fields for selection in this way.
* the second one (for `Age`) shows that arrays can be used instead of dotted names. The elements of array will be applied sequencially to the input. In this particular case the array consists of two strings, and strings are used as the keys to extract values from dictionaries. Therefore, first the `passenger` will be extracted from the top-level dictionary, and then -- `Age` from `passenger`.
* the third (for `ticket.id`) is the only way how we can access the fields with the symbol `.` in name. `FieldGetter` is one of aforementioned small processors: it processes the given object by extracting the element out of the dictionary, or a field from the Python object. 
* the fourth way (for `Sex`) fully represents how selection works under the hood: it is a sequencial application (`Pipeline`) of two `FieldGetters`. So the arrays for `Age` and `ticket.id` will be converted to `Pipeline` under the hood.

The best practice is to use the first method wherever possible, and the third one in other cases.

If you select several fields from the same nested object, please use `with_prefix` method for optimization:


```python
from tg.common.datasets.selectors import Selector, FieldGetter, Pipeline

selector = (Selector()
            .with_prefix('passenger')
            .select('Name','Age','Sex')
            .select(ticket_id = ['ticket',FieldGetter('ticket.id')])
            )
selector(obj)
```




    {'Name': 'Bonnell, Miss. Elizabeth',
     'Age': 58.0,
     'Sex': 'female',
     'ticket_id': '113783'}



`with_prefix` method only affects the `select` that immediately follows it. 

Often, we need to post-process the values. For instance, name by itself is not likely to be feature (and would be GDPR-protected for the actual customers, thus making the entire output dataset GDPR-affected, which is better to avoid). However, we can extract the title from name as it can indeed be a predictor.


```python
import re

def get_title(name):
    title = re.search(' ([A-Za-z]+)\.', name).group().strip()[:-1]
    if title in ['Lady', 'Countess','Capt', 'Col',
                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
        return 'Rare'
    elif title == 'Mlle':
        return 'Miss'
    elif title == 'Ms':
        return 'Miss'
    elif title == 'Mme':
        return 'Mrs'
    else:
        return title

string_size = (Selector()
               .select(['passenger','Name', get_title])
              )

string_size(obj)
```




    {'Name': 'Miss'}



## Adress

Selectors are used to retrieve many fields from the input. Sometimes, we only want to retrieve one field, but with this handy interface. To do that, TG offers `Address` class


```python
from tg.common.datasets.selectors import Address

Address('passenger','Name', get_title)(obj)
```




    'Miss'



Or, the equivalent:


```python
Address.on(obj)('passenger','Name',get_title)
```




    'Miss'



`Address` is extremely useful for exploratory data analysis: by adding and removing arguments, you may move on the object forth and back:


```python
Address.on(obj)(list)
Address.on(obj)('passenger',list)
Address.on(obj)('passenger','Age')
Address.on(obj)('passenger','Age', type)
pass
```

## Ensembles and Pipelines

`Ensemble` and `Pipeline` classes are the machinery behind the `Selector` and `Address`, and ofter they are used directly, alongside with `Selector`. For instance, what if we want to apply multiple featurizers to the same field? We can use `Ensemble` for that:


```python
from tg.common.datasets.selectors import Ensemble

def get_length(s):
    return len(s)

string_size = (Selector()
               .select([
                   'passenger',
                   'Name', 
                   Ensemble(title=get_title, length=get_length)
               ]))


string_size(obj)
```




    {'Name': {'title': 'Miss', 'length': 24}}



`Ensemble` can also be used to combine several selectors together. When your Data Objects are huge and complicated, it makes more sense to write several smaller selectors, instead of writing one that selects all the fields you have. It's easier to read and reuse this way.


```python
ticket_selector = (Selector()
                   .with_prefix('ticket')
                   .select(
                       'fare',
                       'PClass',
                       id = [FieldGetter('ticket.id')]
                   )
)
passenger_selector = (Selector()
                      .with_prefix('passenger')
                      .select(
                          'Sex',
                          'Age',
                          name=['Name', Ensemble(
                              length=get_length,
                              title=get_title
                          )]
                      ))

combined_selector = Ensemble(
    ticket = ticket_selector,
    passenger = passenger_selector,
)
combined_selector(obj)
```

    2022-06-27 19:28:38.748037+00:00 WARNING: Missing field in FieldGetter





    {'ticket': {'id': '113783', 'fare': 26.55, 'PClass': None},
     'passenger': {'name': {'length': 24, 'title': 'Miss'},
      'Sex': 'female',
      'Age': 58.0}}



Pipelines, too, can be used for combination purposes. The typical use case is postprocessing: at the first step, we select fields from the initial object, and then, we want to compute some functions from these fields (e.g., we may want to compute BMI for the person from their weight and height). 

In Titanic example, let's compute a sum of `SibSp` and `Patch` as a new feature, `Relatives`. We will place it into the new `trip_selector` (which is selector, describing the trip in general, rather than the person or the ticket).

For that, we will use `Pipeline`. The arguments of the `Pipeline` are functions, that will be sequencially applied to the input.


```python
def add_relatives_count(d):
    d['Relatives'] = d['SibSp'] + d['Patch']
    return d

trip_selector = Pipeline(
    Selector()
     .select('id')
     .with_prefix('trip')
     .select('Survived','Cabin','Embarked','SibSp','Patch'),
    add_relatives_count
)

trip_selector(obj)
```




    {'id': 12,
     'Survived': 1,
     'Cabin': 'C103',
     'Embarked': 'S',
     'SibSp': 0,
     'Patch': 0,
     'Relatives': 0}



Now we need to do some finishing stitches: 
* for a problemless conversion to dataframe, we need a flat `dict`, not nested. TG has the method for that, `flatten_dict`
* We will also insert the current time as a processing time.


```python
from tg.common.datasets.selectors import flatten_dict
import datetime

def add_meta(obj):
    obj['processed'] = datetime.datetime.now()
    return obj

titanic_selector = Pipeline(
    Ensemble(
        passenger = passenger_selector,
        ticket = ticket_selector,
        trip = trip_selector
    ),
    add_meta,
    flatten_dict
)
titanic_selector(obj)
```

    2022-06-27 19:28:38.768354+00:00 WARNING: Missing field in FieldGetter





    {'passenger_name_length': 24,
     'passenger_name_title': 'Miss',
     'passenger_Sex': 'female',
     'passenger_Age': 58.0,
     'ticket_id': '113783',
     'ticket_fare': 26.55,
     'ticket_PClass': None,
     'trip_id': 12,
     'trip_Survived': 1,
     'trip_Cabin': 'C103',
     'trip_Embarked': 'S',
     'trip_SibSp': 0,
     'trip_Patch': 0,
     'trip_Relatives': 0,
     'processed': datetime.datetime(2022, 6, 27, 21, 28, 38, 769388)}



## Representation

The selectors always keep the internal structure and thus can be analyzed and represented in the different format. The following code demonstrates how this structure can be retrieved. 


```python
from tg.common.datasets.selectors import CombinedSelector
import json

def process_selector(selector):
    if isinstance(selector, CombinedSelector):
        children = selector.get_structure()
        if children is None:
            return selector.__repr__()
        result = {} # {'@type': str(type(selector))}
        for key, value in children.items():
            result[key] = process_selector(value)
        return result
    return selector.__repr__()


representation = process_selector(titanic_selector)

print(json.dumps(representation, indent=1)[:300]+"...")
            
```

    {
     "0": {
      "passenger": {
       "0": {
        "0": {
         "0": "[?passenger]"
        },
        "1": {
         "name": {
          "0": "[?Name]",
          "1": {
           "length": "<function get_length at 0x7fd80a3c2040>",
           "title": "<function get_title at 0x7fd80a428ca0>"
          }
         },
         "Sex": {
          "0": "...


To date, we didn't really find out the format that is both readable and well-representative, so we encourage you to explore and extend the code for representation creation to add the field you need for the effective debugging.

## Error handling

Sometimes selectors throw an error while processing the request. They provide a powerful tracing mechanism to find the cause of error in their complicated structure, as well as in the original piece of data.

Let us create an erroneous object for processing. The `Name` field which is normally a string, will be replaced with integer value.


```python
err_obj = source.get_data().first()
err_obj['passenger']['Name'] = 0
err_obj
```




    {'id': 1,
     'ticket': {'ticket.id': 'A/5 21171', 'fare': 7.25, 'Pclass': 3},
     'passenger': {'Name': 0, 'Sex': 'male', 'Age': 22.0},
     'trip': {'Survived': 0,
      'SibSp': 1,
      'Patch': 0,
      'Cabin': nan,
      'Embarked': 'S'}}




```python
from tg.common.datasets.selectors import SelectorException
exception = None
try:
    titanic_selector(err_obj)
except SelectorException as ex:
    exception = ex
    
print(exception.context.original_object)
print(exception.context.get_data_path())
print(exception.context.get_code_path())

```

    {'id': 1, 'ticket': {'ticket.id': 'A/5 21171', 'fare': 7.25, 'Pclass': 3}, 'passenger': {'Name': 0, 'Sex': 'male', 'Age': 22.0}, 'trip': {'Survived': 0, 'SibSp': 1, 'Patch': 0, 'Cabin': nan, 'Embarked': 'S'}}
    [?passenger].[?Name]
    /0/passenger/0/1/name/1/length:get_length


Selectors are usually applied to the long sequences of data which may not be reproducible. It is therefore wise to cover your featurization with try-except block and store the exception on the hard drive, so you could later build a test case with `original_object`.

`get_data_path()` returns the string representation of path inside data where the error has occured: somewhere around `obj['passenger']['Name']`. Symbol `?` means that these fields are optional, and _all fields_ are optional by default. If you want the selector that raises exception when the field does not exist, pass the `True` argument to the constructor of the `Selector`.

`get_code_path()` describes where the error occured within the hierarchy of selectors. By looking at this string, we can easily figure out that error occured somewhere around processing `name` with `get_length` method. If the deeper analysis is required, we may use the `representation` object we have previously built:


```python
representation[0]['passenger'][0]
```




    {0: {0: '[?passenger]'},
     1: {'name': {0: '[?Name]',
       1: {'length': '<function get_length at 0x7fd80a3c2040>',
        'title': '<function get_title at 0x7fd80a428ca0>'}},
      'Sex': {0: '[?Sex]'},
      'Age': {0: '[?Age]'}}}



Here we input the beginning of `get_code_path()` output and see the closer surroundings of the error.

## List featurization

In this section we will consider building features for a list of objects. This use case is rather rare, the examples is, for instance, building the features for a customer, basing on the articles he have purchased in the past.

In the Titanic setup, imagine that:
1. Our task is actually produce feature for cabins, not for passengers.
2. Our DOF is also flow of cabins, not passengers.

So one object looks like this:


```python
cabin_obj = source.get_data().where(lambda z: z['trip']['Cabin']=='F2').to_list()
cabin_obj
```




    [{'id': 149,
      'ticket': {'ticket.id': '230080', 'fare': 26.0, 'Pclass': 2},
      'passenger': {'Name': 'Navratil, Mr. Michel ("Louis M Hoffman")',
       'Sex': 'male',
       'Age': 36.5},
      'trip': {'Survived': 0,
       'SibSp': 0,
       'Patch': 2,
       'Cabin': 'F2',
       'Embarked': 'S'}},
     {'id': 194,
      'ticket': {'ticket.id': '230080', 'fare': 26.0, 'Pclass': 2},
      'passenger': {'Name': 'Navratil, Master. Michel M',
       'Sex': 'male',
       'Age': 3.0},
      'trip': {'Survived': 1,
       'SibSp': 1,
       'Patch': 1,
       'Cabin': 'F2',
       'Embarked': 'S'}},
     {'id': 341,
      'ticket': {'ticket.id': '230080', 'fare': 26.0, 'Pclass': 2},
      'passenger': {'Name': 'Navratil, Master. Edmond Roger',
       'Sex': 'male',
       'Age': 2.0},
      'trip': {'Survived': 1,
       'SibSp': 1,
       'Patch': 1,
       'Cabin': 'F2',
       'Embarked': 'S'}}]



We want to build the following features for this `cabin_obj`: the average fare and age of the passengers. 

So, to build such aggregated selectors, following practice is recommended:
1. build a `Selector` that selects the fields, and apply it to the list, building list of dictionaries
2. convert list of dictionaries into dictionary of lists
3. apply averager to each list.

Let's first do it step-by-step. `Listwise` applies arbitrary function (e.g., your selector) to the elements of the list:


```python
from tg.common.datasets.selectors import Listwise, Dictwise, transpose_list_of_dicts_to_dict_of_lists

cabin_features_selector = (Selector()
                           .select('passenger.Age','ticket.fare')
                          )
list_of_dicts = Listwise(cabin_features_selector)(cabin_obj)
list_of_dicts
```




    [{'Age': 36.5, 'fare': 26.0},
     {'Age': 3.0, 'fare': 26.0},
     {'Age': 2.0, 'fare': 26.0}]



`transpose_list_of_dicts_to_dict_of_lists` makes the "transposition" of list of dicts into dict of lists.


```python
dict_of_lists = transpose_list_of_dicts_to_dict_of_lists(list_of_dicts)
dict_of_lists
```




    {'Age': [36.5, 3.0, 2.0], 'fare': [26.0, 26.0, 26.0]}



Finally, `Dictwise` applies function to the elements of dictionary


```python
import numpy as np

Dictwise(np.mean)(dict_of_lists)
```




    {'Age': 13.833333333333334, 'fare': 26.0}



If you need a more complicated logic, such as applying different functions to different fields, you will need to extend `Dictwise` class.

All we have to do now is to assemble it to the pipeline. Since in our use cases we have used this pipeline several times, it's standardized in the following class:


```python
from tg.common.datasets.selectors import ListFeaturizer

selector = ListFeaturizer(cabin_features_selector, np.mean)
selector(cabin_obj)
```




    {'Age': 13.833333333333334, 'fare': 26.0}



### Quick dataset creations

The combination of `DataSource` and `Featurizer` allows you to quickly build the tidy dataset:


```python
source.get_data().take(3).select(titanic_selector).to_dataframe()
```

    2022-06-27 19:28:38.853372+00:00 WARNING: Missing field in FieldGetter
    2022-06-27 19:28:38.854069+00:00 WARNING: Missing field in FieldGetter
    2022-06-27 19:28:38.854707+00:00 WARNING: Missing field in FieldGetter





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>passenger_name_length</th>
      <th>passenger_name_title</th>
      <th>passenger_Sex</th>
      <th>passenger_Age</th>
      <th>ticket_id</th>
      <th>ticket_fare</th>
      <th>ticket_PClass</th>
      <th>trip_id</th>
      <th>trip_Survived</th>
      <th>trip_Cabin</th>
      <th>trip_Embarked</th>
      <th>trip_SibSp</th>
      <th>trip_Patch</th>
      <th>trip_Relatives</th>
      <th>processed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>Mr</td>
      <td>male</td>
      <td>22.0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>None</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2022-06-27 21:28:38.853902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51</td>
      <td>Mrs</td>
      <td>female</td>
      <td>38.0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>None</td>
      <td>2</td>
      <td>1</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2022-06-27 21:28:38.854524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>Miss</td>
      <td>female</td>
      <td>26.0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>None</td>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2022-06-27 21:28:38.855246</td>
    </tr>
  </tbody>
</table>
</div>



If your selector is small, you may also define `Selector` on the fly:


```python
(source
 .get_data()
 .take(3)
 .select(Selector().select('passenger.Name','trip.Survived'))
 .to_dataframe()
)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Summary

In this demo, we have presented how you may use `Selector` and other classes to convert complex, hierarchical objects into rows in the tidy dataset. 

