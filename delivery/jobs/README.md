# 4.2. Deliverable Jobs (tg.common.delivery.jobs)

## SSH/Docker routine

Fortunately, you don't really need to do packaging or containering yourself, because we have a higher level level interfaces to do that, which is `Routine` classes. 

Let's create a job that we are going to package:


```python
from tg.common.datasets.featurization import FeaturizationJob, DataframeFeaturizer
from tg.common.datasets.selectors import Selector
from tg.common.datasets.access import MockDfDataSource
from tg.common import MemoryFileSyncer
import pandas as pd

mem = MemoryFileSyncer()

job = FeaturizationJob(
    name = 'job',
    version = 'v1',
    source = MockDfDataSource(pd.read_csv('titanic.csv')),
    featurizers = {
        'passengers': DataframeFeaturizer(row_selector = Selector.identity)
    },
    syncer = mem,
    location = './temp/featurization_job'
)


```

Then, `SSHDockerJobRoutine` allows you to execute your jobs in the docker at the remote server to which you have ssh access.


```python
from tg.common.delivery.jobs import SSHDockerJobRoutine, DockerOptions
from tg.common.delivery.packaging import FakeContainerHandler

routine = SSHDockerJobRoutine(
    job = job,
    remote_host_address=None,
    remote_host_user=None,
    handler_factory = FakeContainerHandler.Factory(),
    options = DockerOptions(propagate_environmental_variables=[])
)
```

Most of the fields are specified to None, because we are not going to actually start the remote job with this notebook. `remote_host_address` and `remote_host_user` arguments are self-explainatory. 

As for `handler_factory`, this argument must be set to one of the factories that generate `ContainerHandler`. This `ContainerHandler` class must define a remote image name and tag, and perform push operations. `ContainerHandlers` are not included to the Training Grounds core, as they usually have some company-specific code.

`SSHDockerJobRoutine` has methods of running your code for debugging.

Using the `.attached` accesor, we can run job in the same Python process that your current code is executed. This is, of course, the fastest way to do that, and therefore it's preferrable to use this to debug for typos, wrong logic, etc.


```python
routine.attached.execute()
```

    2022-06-27 19:40:31.016162+00:00 INFO: Featurization Job job at version v1 has started
    2022-06-27 19:40:31.017706+00:00 INFO: Fetching data
    2022-06-27 19:40:31.075090+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:40:31.098569+00:00 INFO: Uploading data
    2022-06-27 19:40:31.099733+00:00 INFO: Featurization job completed


The `.local` accessor builds package and container, then executes the container locally. This step allows debugging the following things:

* If your job is serializable. This is usually achievable by not using `lambda` syntax.
* If all the code the job uses is located inside the TG folder, and if all the references are relative. If something is wrong, you will see the import error.
* If the environmental variables are carried to docker correctly. 
* If you have sufficient permissions to start docker
* etc.


This step allows you to check the deliverability of your work. As the output is quite big, we will remove it for better readability.


```python
from IPython.display import clear_output

routine.local.execute()
clear_output()
```

You can retrieve logs from the container with the following useful method. Note that logs printed via `logging` are placed in stderr instead of strdout.


```python
output, errors = routine.local.get_logs()
print(output)
```

    2022-06-27 19:40:54.883388+00:00 INFO: Welcome to Training Grounds. This is Job execution via Docker/SSH
    2022-06-27 19:40:54.930855+00:00 INFO: Executing job job version v1
    2022-06-27 19:40:54.931020+00:00 INFO: Featurization Job job at version v1 has started
    2022-06-27 19:40:54.931552+00:00 INFO: Fetching data
    2022-06-27 19:40:55.228919+00:00 INFO: Data fetched, finalizing
    2022-06-27 19:40:55.310273+00:00 INFO: Uploading data
    2022-06-27 19:40:55.312840+00:00 INFO: Featurization job completed
    2022-06-27 19:40:55.313731+00:00 INFO: Job completed
    


`routine.remote` has the same interface as `routine.local`, and will run the container at the remote machine. The only problems you should have at these stage are permissions:
* to push to your docker registry
* to connect to the remote machine via SSH
* to execute `docker run` at the remote machine

## Summary 

In this demo, we delivered the job to the remote server and executed it there. That concludes the featurization-related part of the Training Grounds.

Note that the packaging and containering techniques are not specific for the featurization, and can process any code. In the subsequent demos, the same techniques will be applied to run the training on the remote server as well.
