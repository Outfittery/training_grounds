from unittest import TestCase
from tg.common.delivery.jobs import SSHDockerJobRoutine, JobExecutor, DeliverableJob, DockerOptions
from tg.common.test_common.test_delivery.test_jobs.job import Job
import logging

logger = logging.getLogger()


def make_test(executor_selector):
    routine = SSHDockerJobRoutine(
        Job(),
        '',
        '',
        '',
        DockerOptions()
    )
    executor = executor_selector(routine)  # type: JobExecutor
    executor.execute()
    return executor


class HetznerTestCase(TestCase):
    def test_attached(self):
        make_test(lambda z: z.attached)

    def test_local(self):
        executor = make_test(lambda z: z.local)
        out, err = executor.get_logs()
        self.assertIn('Job::run method', err)
