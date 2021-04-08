from .....common.delivery.jobs import DeliverableJob
import logging

logger = logging.getLogger()

class Job(DeliverableJob):
    def get_name_and_version(self):
        return 'testjob','v0'

    def run(self):
        logger.info("Job::run method is called")
