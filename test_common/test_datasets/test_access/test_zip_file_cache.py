from unittest import TestCase
from tg.common.datasets.access import *
from yo_fluq_ds import Query, FileIO
from pathlib import Path
import os

class ZipFileCacheTestCase(TestCase):
    def test_zip_file(self):
        src = Query.en(range(10))
        path = Path(__file__).parent.joinpath('test_cache')
        cache = ZippedFileDataSource(path,buffer_size=4)

        self.assertEqual(False, cache.is_available())
        cache.cache_from(src,7)
        self.assertEqual(True, cache.is_available())

        self.assertEqual(
            "7",
            FileIO.read_text(path.__str__()+'.pkllines.zip.length')
        )

        stored = Query.file.zipped_folder(path.__str__()+'.pkllines.zip').to_dictionary()
        self.assertEqual(2,len(stored))
        self.assertListEqual([0,1,2,3], stored['0'])
        self.assertListEqual([4,5,6], stored['1'])

        result = cache.get_data().to_list()
        self.assertListEqual(list(range(7)),result)

        os.unlink(path.__str__()+'.pkllines.zip.length')
        os.unlink(path.__str__() + '.pkllines.zip')