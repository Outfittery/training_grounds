from unittest import TestCase
from tg.common.delivery.training import *
from tg.common.test_common.test_delivery.test_training.task import create_dataset_files, create_task
from tg.common import Loc
from sklearn.metrics import accuracy_score

dataset_folder = Loc.temp_path/'tests/training_ssh_docker_delivery_test_case/datasets'
local_folder = Loc.temp_path/'tests/training_ssh_docker_delivery_test_case/home'
create_dataset_files(dataset_folder, 'test')

docker_routine = SSHDockerTrainingRoutine(
    dataset_folder,
    'testcase',
    local_folder,
    None, None, None, None, None
)

class TrainingDeliveryTestCase(TestCase):
    def make_test(self, selector):
        executor = selector(docker_routine)
        id = executor.execute(create_task(2000), 'test')
        result = executor.get_result(id)
        df = result.unpickle('runs/0/result_df')
        self.assertLess(0.9, accuracy_score(df.true, df.predicted))
        model = result.unpickle('runs/0/model')
        self.assertEqual(2000, model.max_iter)

    def test_attached(self):
        self.make_test(lambda z: z.attached)

    def test_local(self):
        self.make_test(lambda z: z.local)



