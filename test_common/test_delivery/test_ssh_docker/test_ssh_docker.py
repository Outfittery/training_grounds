from unittest import TestCase
from tg.common.delivery.delivery import Packaging, Containering
from tg.common.delivery.ssh_docker import *
from tg.common.delivery.delivery.example_job import ExampleJob

def create_config():
    packaging = Packaging(
        'test_job',
        '1',
        dict(job = ExampleJob())
    )
    containering = Containering.from_packaging(packaging)
    return SSHDockerConfig(packaging, containering, SSHDockerOptions())

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

