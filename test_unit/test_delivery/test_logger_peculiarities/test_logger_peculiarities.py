from unittest import TestCase
from tg.common.test_unit.test_delivery.test_logger_peculiarities.job import JobLogProblem

from tg.common.delivery.delivery import Container
from tg.common.delivery.ssh_docker import SSHDockerConfig, SSHDockerOptions, SSHLocalExecutor

def create_config(job):
    container = Container(
        'test_job',
        '1',
        job,
        ['boto3', 'pandas', 'yo_fluq_ds'],
        ['tg'],
        None
    )
    return SSHDockerConfig(container, SSHDockerOptions())

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


