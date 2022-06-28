from yo_fluq_ds import Query
from enum import Enum

class Autonamer:
    def __init__(self, build_method):
        self.build_method = build_method

    def build_tasks(self, **kwargs):
        calls = Query.combinatorics.grid(**kwargs)
        result = []
        for call in calls:
            parts = []
            for key, value in call.items():
                prefix = ''.join([c[0].upper() for c in key.split('_')])
                v = ''
                if isinstance(value, int):
                    if value > 1000000:
                        v = str(value // 1000000) + "M"
                    elif value > 1000:
                        v = str(value // 1000) + "K"
                    else:
                        v = str(value)
                elif isinstance(value, list):
                    v = '-'.join(str(c) for c in value)
                elif isinstance(value, float):
                    v = str(value).replace('.', '')
                elif isinstance(value, Enum):
                    v = value.name
                else:
                    v = str(value)
                parts.append(prefix + v)
            name_suffix = '-'.join(parts)
            task = self.build_method(**call)
            task.info['name'] += name_suffix
            result.append(task)
        return result

