# 1.1. Common features (tg.common._common)

`tg.common` defines a few auxiliary classes, that are used through the whole framework.

To prevent circular dependencies, the code is actually defined in `tg.common._common`. Not everything is exposed to the `tg.common`, as we want to limit the amount of universally exposed classes to the minimum.

## Logger

`Logger` is a slim wrapper over standard `logging` module, designed to augment the logging messages with "keys" that additionally describe the message. The main reason for this is integration with i.e. Kibana. 


These fields belong to one of the categories:
  * Automatic: code file, line, exception type, value and stacktrace
  * Base: the name of the service, version, etc.
  * Session: the user-defined keys.

Logger will work without any additional initialization, just by importing:


```python
from tg.common import Logger

Logger.info('Message with default logger')
```

    2022-08-09 09:25:48.886388+00:00 INFO: Message with default logger



```python
Logger.initialize_kibana()
Logger.info('Message with Kibana logger')
```

    {"@timestamp": "2022-08-09 09:25:48.896592+00:00", "message": "Message with Kibana logger", "levelname": "INFO", "logger": "tg", "path": "/tmp/ipykernel_10321/2907404717.py", "path_line": 2}


As said before, you may define a custom session keys:


```python
Logger.push_keys(test_key='test')
Logger.info('Message with a key')
Logger.clear_keys()
Logger.info('Message without a key')
```

    {"@timestamp": "2022-08-09 09:25:48.905318+00:00", "message": "Message with a key", "levelname": "INFO", "logger": "tg", "path": "/tmp/ipykernel_10321/71300885.py", "path_line": 2, "test_key": "test"}
    {"@timestamp": "2022-08-09 09:25:48.906591+00:00", "message": "Message without a key", "levelname": "INFO", "logger": "tg", "path": "/tmp/ipykernel_10321/71300885.py", "path_line": 4}


If exception information is available, it will be put in the keys:


```python
try:
    raise ValueError('Error')
except: 
    Logger.error('Error')
```

    {"@timestamp": "2022-08-09 09:25:48.916677+00:00", "message": "Error", "levelname": "ERROR", "logger": "tg", "path": "/tmp/ipykernel_10321/1975102656.py", "path_line": 4, "exception_type": "<class 'ValueError'>", "exception_value": "Error", "exception_details": "Traceback (most recent call last):\n  File \"/tmp/ipykernel_10321/1975102656.py\", line 2, in <module>\n    raise ValueError('Error')\nValueError: Error\n"}


To change the default way of logging, inherit `tg.common._common.logger.LoggerRoot` in `tg.your_project`, and then import from there. It will also affect all the logging within `TG` framework.

## Loc

In various situations, `TG` stores intermediate or cache files on your machine, typically in:
* `/temp` folder, these files do not require large efforts to create and thus you can delete it as often as you like.
* `/data-cache`: datasets and data downloads are stored there, and so restoring these files may take awhile.

To access these files:


```python
from tg.common import Loc
from yo_fluq_ds import FileIO

FileIO.write_text('test', Loc.temp_path/'test.txt')
FileIO.read_text(Loc.temp_path/'test.txt')
```




    'test'


