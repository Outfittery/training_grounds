from unittest import TestCase
from tg.common.datasets.access import *
from yo_fluq_ds import Query, FileIO, KeyValuePair
from pathlib import Path
import os
import shutil
from tg.common import Loc
import pickle


class ZipFileCacheTestCase(TestCase):
    def test_zip_file(self):
        src = Query.en(range(10))
        path = Loc.temp_path / 'tests/data_source_zip_cache/data'
        shutil.rmtree(path.parent, ignore_errors=True)
        os.makedirs(path.parent)

        cache = ZippedFileDataSource(path, buffer_size=4)

        self.assertEqual(False, cache.is_available())
        cache.cache_from(src, 7)
        self.assertEqual(True, cache.is_available())

        stored = Query.file.zipped_folder(path.__str__(), parser=lambda z: z).to_dictionary()
        self.assertEqual(3, len(stored))
        self.assertEqual(7, int(stored['length'].decode('utf-8')))
        self.assertListEqual([0, 1, 2, 3], pickle.loads(stored['data/0']))
        self.assertListEqual([4, 5, 6], pickle.loads(stored['data/1']))

        result = cache.get_data()
        self.assertEqual(7, result.length)
        self.assertListEqual(list(range(7)), list(result))

    def test_zip_file_on_old_format(self):
        data = [[0, 1, 2], [3, 4, 5]]
        path = Loc.temp_path / 'tests/data_source_old_format/data'
        os.makedirs(path.parent, exist_ok=True)
        Query.en(data).with_indices().select(lambda z: KeyValuePair(str(z.key), z.value)).to_zip_folder(path)

        cache = ZippedFileDataSource(path)
        result = cache.get_data()
        self.assertIsNone(result.length)
        self.assertListEqual([0, 1, 2, 3, 4, 5], result.to_list())
