import logging

from ...delivery.packaging import EntryPoint



logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO)  # TODO: because kblog does not show INFO message when run locally
logger = logging.getLogger()


def execute_featurization_job(entry: EntryPoint):
    logger.info("Welcome to Training Grounds. This is Job execution via Docker/SSH")
    # for key, value in os.environ.items(): logger.info(f"{key}:>>{value}<<")
    job = entry.load_resource('job')
    name, version = job.get_name_and_version()
    logger.info(f"Executing job {name} version {version}")
    job.run()
    logger.info("Job completed")
