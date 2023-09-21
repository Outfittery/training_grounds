from ..data_bundle import DataBundle
from pathlib import Path

class TestBundles:
    @staticmethod
    def get_test_bundle():
        return Path(__file__).parent/'tsa-test.zip'

    @staticmethod
    def get_test_2_bundle():
        return Path(__file__).parent / 'tsa-test-2.zip'
