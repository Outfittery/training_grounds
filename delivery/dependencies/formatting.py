from yo_fluq_ds import *

if __name__ == '__main__':
    deps = FileIO.read_json('default_dependencies.json')
    print(' '.join(deps['default']))