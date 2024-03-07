from unittest import TestCase

import pandas as pd
from pandas.testing import assert_frame_equal

from tg.common import Loc
from tg.common.data import SQLAccessor
from tg.common.outfittery.data import SQLInserter


class SqlInserterTestCase(TestCase):
    def setUp(self) -> None:
        dir = Loc.temp_path / "tests"
        dir.mkdir(parents=True, exist_ok=True)
        path = dir / "sqlaccessor"
        if path.is_file():
            path.unlink()
        self.dsn = f"sqlite:///{path}"

    def test_insert_new_table(self):
        input_df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]}, index=pd.Index([10, 11, 12], name="index")
        )
        sql_inserter = SQLInserter(self.dsn)
        sql_inserter.insert_data(input_df, "test")
        output_df = (
            SQLAccessor("SELECT * from test", self.dsn).get_data().set_index("index")
        )
        assert_frame_equal(input_df, output_df)

    def test_insert_existing_table(self):
        initial_df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]}, index=pd.Index([10, 11, 12], name="index")
        )
        sql_inserter = SQLInserter(self.dsn)
        sql_inserter.insert_data(initial_df, "test")
        append_df = pd.DataFrame(
            {"a": [4, 5, 6], "b": [7, 8, 9]}, index=pd.Index([13, 14, 15], name="index")
        )
        sql_inserter.insert_data(append_df, "test")
        input_df = pd.concat([initial_df, append_df])
        output_df = (
            SQLAccessor("SELECT * from test", self.dsn).get_data().set_index("index")
        )
        assert_frame_equal(input_df, output_df)

    def test_insert_with_chunks(self):
        input_df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]}, index=pd.Index([10, 11, 12], name="index")
        )
        sql_inserter = SQLInserter(self.dsn, chunk_size=2)
        sql_inserter.insert_data(input_df, "test")
        output_df = (
            SQLAccessor("SELECT * from test", self.dsn).get_data().set_index("index")
        )
        assert_frame_equal(input_df, output_df)

    def test_insert_no_index(self):
        input_df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]}, index=pd.Index([10, 11, 12], name="index")
        )
        sql_inserter = SQLInserter(self.dsn, include_index=False)
        sql_inserter.insert_data(input_df, "test")
        output_df = SQLAccessor("SELECT * from test", self.dsn).get_data()
        assert_frame_equal(input_df.reset_index(drop=True), output_df)
