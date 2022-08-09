from pathlib import Path
import sys
import json

if __name__ == '__main__':
    path_to_dependencies_file = Path(__file__).parent/'default_dependencies.json'
    profile = 'default'
    #TODO alter those if command line arguments are provided

    with open(path_to_dependencies_file,'r') as file:
        dep_list = json.load(file)[profile]

    print(json.dumps(dep_list))



