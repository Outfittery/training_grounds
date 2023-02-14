from unittest import TestCase
from tg.common.test_common.test_delivery.test_logger_peculiarities.job import JobLogProblem
from tg.common.test_common.test_delivery.test_logger_peculiarities.more_complicated_job import JobInstallingPackage

from tg.common.delivery.delivery import Packaging, Containering
from tg.common.delivery.ssh_docker import SSHDockerConfig, SSHDockerOptions, SSHLocalExecutor

def create_config(job):
    packaging = Packaging(
        'test_job',
        '1',
        dict(job = job)
    )
    containering = Containering.from_packaging(packaging)
    return SSHDockerConfig(packaging, containering, SSHDockerOptions())

class LoggerPeculiaritiesTestCase(TestCase):
    def count_all(self, substr, s):
        return sum([s[i:].startswith(substr) for i in range(len(s))])

    def test_log_problem(self):
        config = create_config(JobLogProblem())
        loc = SSHLocalExecutor(config)
        loc.execute()
        logs = loc.get_logs()
        self.assertEqual(1, self.count_all('SUCCESS', logs[0]))
        self.assertEqual(1, self.count_all('DONE. Exiting Training Grounds.', logs[0]))


    def test_complicated_log_problem(self):
        config = create_config(JobInstallingPackage())
        loc = SSHLocalExecutor(config)
        loc.execute()
        logs = loc.get_logs()
        self.assertEqual(1, self.count_all('I am in package', logs[0]))
        self.assertEqual(1, self.count_all('Success', logs[0]))
        self.assertEqual(1, self.count_all('DONE. Exiting Training Grounds.', logs[0]))

