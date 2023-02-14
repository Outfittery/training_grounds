
PACKAGE_MANFEST_TEMPLATE = '''
include {module_name}/resources/*
recursive-include {module_name} *.yml *.json *.py *.rst
include properties.json
'''


PACKAGE_SETUP_PY_TEMPLATE = '''
from setuptools import setup, find_packages


setup(name='{name}',
      version='{version}',
      description='',
      packages=find_packages(),
      install_requires=[
          {dependencies}
      ],
      include_package_data = True,
      zip_safe=False)
'''


PACKAGE_INIT_TEMPLATE = '''
from .tg.common.delivery.delivery.entry_point import EntryPoint
from pathlib import Path

Entry = EntryPoint(
    '{name}',
    '{version}',
    '{module_name}',
    '{tg_import_path}',
    '{original_tg_import_path}',
    Path(__file__).parent.joinpath('resources')
)
'''

DOCKER_RUN_PY_TEMPLATE = '''
from {module} import Entry
from {module}.tg.common import Logger
import traceback

Logger.initialize_default()
Logger.info('Welcome to Training Grounds!')

Logger.info('Loading job')
try:
    job = Entry.load_resource('job')
    Logger.info('Job of type '+str(type(job))+' is loaded')
except:
    tb = traceback.format_exc()
    Logger.error('Job is NOT loaded')
    Logger.error(tb)
    raise
    

def run(m):
    try:
        m()
        Logger.info('Job has exited successfully')
    except:
        tb = traceback.format_exc()
        Logger.error('Job has NOT exited sucessfully')
        Logger.error(tb)
        raise
    

if callable(job):
    Logger.info('Job is callable, calling directly')
    run(job)
elif hasattr(job, 'run'):
    Logger.info('Job has `run` attribute')
    if hasattr(job, 'set_calling_entry_point'):
        job.set_calling_entry_point(Entry)
    run(job.run)
else:
    raise ValueError('Job is not callable and does not have `run` attributes')

Logger.info('DONE. Exiting Training Grounds.')
'''

DOCKERFILE_TEMPLATE = '''FROM python:{python_version}

{install_libraries}

COPY . /job

WORKDIR /job

COPY {package_filename} package.tar.gz

RUN pip install package.tar.gz

CMD ["python3","/job/{run_file_name}"]
'''