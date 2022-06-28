from ...delivery.packaging import EntryPoint
from ..._common import Logger


Logger.initialize_default()


def execute_featurization_job(entry: EntryPoint):
    Logger.info("Welcome to Training Grounds. This is Job execution via Docker/SSH")
    # for key, value in os.environ.items(): logger.info(f"{key}:>>{value}<<")
    job = entry.load_resource('job')
    name, version = job.get_name_and_version()
    Logger.info(f"Executing job {name} version {version}")
    job.run()
    Logger.info("Job completed")
