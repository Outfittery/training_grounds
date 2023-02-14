# 4.2. Delivery via SSH (tg.common.delivery.ssh_docker)

One of the scenarios for delivery is to start the job at a remote docker server via SSH. Training Grounds contain several classes that facilitate this process.

First, we have `SSHDockerOptions`. This class contains the settings on how to run the job in the docker container: which environmental variables are to propagate from the local machine to the remote one, memory and CPU limits. 

Second, we have `SSHDockerConfig`: a comprehensive configuration for the procedure that contains `Packaging`, `Containering` and `SSHDockerOptions`, as well as the address of the remote host and the username. 

This `SSHDockerConfig` is an argument for `Executors`, which actually execute the job. We have:
  
  * `AttachedExecutor` that runs the job in the same process, without any docker at all.
  * `LocalExecutor` that runs the job in the local docker
  * `RemoteExecutor` that runs the job in the remote docker, using SSH.
  
These three executors help to debug the job. First, we can run `AttachedExecutor` to make sure that the job itself works. By running it in `LocalExecutor` we make sure that packaging and containering work, e.g.:

* Your job is serializable. This is usually achievable by not using lambda syntax.
* All the code the job uses is located inside the TG folder, and if all the references are relative. If something is wrong, you will see the import error.
* If the environmental variables are carried to docker correctly.
* If you have sufficient permissions to start docker

Finally, `RemoteExecutor` will peform the same functionatily remotely.  The only problems you should have at these stage are permissions:

  * to push to your docker registry
  * to connect to the remote machine via SSH
  * to execute docker run at the remote machine
  
The best way to actually use this code in your project is to write a class, e.g., `SSHDockerRoutine`, in the following way:


```python
from tg.common.delivery.ssh_docker import (SSHAttachedExecutor, SSHLocalExecutor, SSHRemoteExecutor, 
                                           SSHDockerOptions, SSHDockerConfig)
from tg.common.delivery.delivery import Packaging, Containering

variable_name = 'EXAMPLE_VARIABLE'

class SSHDockerRoutine:
    def __init__(self, job):
        self.job = job
        name = type(job).__name__.lower()
        packaging = Packaging(name, '0.0.0', dict(job=job))
        packaging.silent = True
        containering = Containering.from_packaging(packaging)
        containering.silent = True
        options = SSHDockerOptions([variable_name])
        self.config =  SSHDockerConfig(packaging, containering, options, None, None)

    def attached(self):
        return SSHAttachedExecutor(self.config)

    def local(self):
        return SSHLocalExecutor(self.config)
    
        
```

In the `_create_config` method you can place all the logic regarding dependencies, secrets, etc. After this, you can simply use `SSH3DockerRoutine` to run the job remotely. 

Let's run the `ExampleJob` with attached executor:


```python
from tg.common.delivery.delivery.example_job import ExampleJob
import os

os.environ[variable_name] = 'TEST'

job = ExampleJob([variable_name])
routine = SSHDockerRoutine(job)
routine.attached().execute()
```

    2022-12-28 14:25:11.493742 INFO: Variable EXAMPLE_VARIABLE is found: True
    2022-12-28 14:25:11.494948 INFO: SUCCESS


And now, with local:


```python
routine.local().execute()
```

    warning: no files found matching '*.yml' under directory 'examplejob__0_0_0'
    warning: no files found matching '*.rst' under directory 'examplejob__0_0_0'
    warning: sdist: standard file not found: should have one of README, README.rst, README.txt, README.md
    


    sha256:6a9d1deca68272221c3ac4154fe538a0504f2ce9c5c47940dce82b3a8c71024b
    2022-12-28 13:25:22.692513 INFO: Welcome to Training Grounds!
    2022-12-28 13:25:22.692604 INFO: Loading job
    2022-12-28 13:25:22.693027 INFO: Job of type <class 'examplejob__0_0_0.tg.common.delivery.delivery.example_job.ExampleJob'> is loaded
    2022-12-28 13:25:22.693102 INFO: Job has `run` attribute
    2022-12-28 13:25:22.693194 INFO: Variable EXAMPLE_VARIABLE is found: True
    2022-12-28 13:25:22.693262 INFO: SUCCESS
    2022-12-28 13:25:22.693328 INFO: Job has exited successfully
    2022-12-28 13:25:22.693394 INFO: DONE. Exiting Training Grounds.


As we can see, the environment variable was sucessfully transferred from the notebook environment to the docker's environment.
