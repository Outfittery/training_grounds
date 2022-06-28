from .....common.delivery.jobs import DeliverableJob
from .....common import Logger

class Job(DeliverableJob):
    def get_name_and_version(self):
        return 'testjob','v0'

    def run(self):
        Logger.info("Job::run method is called")
