import sqlite3
from unittest import TestCase
from tg.common import Loc
from tg.common.data import SQLAccessor
import pandas as pd
from sqlalchemy import sql


class SqlAccessorTestCase(TestCase):
    def setUp(self) -> None:
        path = Loc.temp_path/'tests/sqlaccessor'
        if path.is_file():
            path.unlink()
        db = sqlite3.connect(path)
        cursor = db.cursor()
        cursor.execute('CREATE TABLE test (a int, b int)')
        for i in range(100):
            cursor.execute(f'INSERT INTO test VALUES ({i}, {i+1})')
        db.commit()
        self.dsn = f'sqlite:////{path}'

    def test_reading(self):
        for query in ['select * from test', sql.select(["*"]).select_from(sql.table('test'))]:
            with self.subTest(str(type(query))):
                df = SQLAccessor(query, self.dsn).get_data()
                self.assertEqual(100, df.shape[0])

    def test_chunk(self):
        for query in ['select * from test', sql.select(["*"]).select_from(sql.table('test'))]:
            with self.subTest(str(type(query))):
                dfs = SQLAccessor(query, self.dsn, chunk_size=20).get_data_flow().to_list()
                self.assertEqual(5, len(dfs))
                for df in dfs:
                    self.assertEqual(20, df.shape[0])

    def test_correct_chunk_length_determination_1(self):
        for query in ['select * from test where a<80', sql.select(["*"]).select_from(sql.table('test')).where(sql.column('a')<80)]:
            with self.subTest(str(type(query))):
                dfs = SQLAccessor(query, self.dsn, chunk_size=20).get_data_flow()
                self.assertEqual(4, dfs.length)
                df = pd.concat(dfs.to_list())
                self.assertListEqual(list(range(80)), list(df.a))

    def test_correct_chunk_length_determination_2(self):
        for query in ['select * from test where a<81', sql.select(["*"]).select_from(sql.table('test')).where(sql.column('a')<81)]:
            with self.subTest(str(type(query))):
                dfs = SQLAccessor(query, self.dsn, chunk_size=20).get_data_flow()
                self.assertEqual(5, dfs.length)
                df = pd.concat(dfs.to_list())
                self.assertListEqual(list(range(81)), list(df.a))

    def test_caching_correct_str(self):
        ac1 = SQLAccessor("select * from test where a<80", self.dsn)
        ac2 = SQLAccessor("select * from test where a<80", self.dsn)
        ac3 = SQLAccessor("select * from test where a<81", self.dsn)
        self.assertEqual(ac1.get_name(), ac2.get_name())
        self.assertNotEqual(ac1.get_name(), ac3.get_name())


    def test_caching_correct_select(self):
        ac1 = SQLAccessor(sql.select(["*"]).select_from(sql.table('test')).where(sql.column('a')<81), self.dsn)
        ac2 = SQLAccessor(sql.select(["*"]).select_from(sql.table('test')).where(sql.column('a') < 81), self.dsn)
        ac3 = SQLAccessor(sql.select(["*"]).select_from(sql.table('test')).where(sql.column('a')<82), self.dsn)
        self.assertEqual(ac1.get_name(), ac2.get_name())
        self.assertNotEqual(ac1.get_name(), ac3.get_name())
