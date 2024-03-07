from unittest import TestCase
from tg.common.delivery.delivery import Container
from tg.common.delivery.ssh_docker import *
from tg.common.delivery.delivery.example_job import ExampleJob


def create_config():
    container = Container(
        'test_job',
        '1',
        ExampleJob(),
        ['boto3','pandas','yo_fluq_ds'],
        ['tg'],
        None
    )
    return SSHDockerConfig(container, SSHDockerOptions())

class SSHDockerTestCase(TestCase):
    def test_attached(self):
        config = create_config()
        SSHAttachedExecutor(config).execute()

    def test_local(self):
        config = create_config()
        loc = SSHLocalExecutor(config)
        loc.execute()
        logs = loc.get_logs()
        self.assertIn('SUCCESS', logs[0].split('\n')[-4])

