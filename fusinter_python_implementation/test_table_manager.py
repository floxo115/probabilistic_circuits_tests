import numpy as np
import pytest

from datasets import paper_dataset
from .table_manager import TableManager


class TestTableManager:
    @pytest.mark.parametrize("data_x, data_y, init_splits, expected", [
        [
            paper_dataset.paper_dataset_x,
            paper_dataset.paper_dataset_y,
            np.array([2, 3, 13, 14, 15, 16, 17, 18, 19, 20, 23, 37, 38, 39, 40], dtype=np.float64),
            np.array([1, 0, 26, 2, 3, 0, 3, 0, 2, 0, 5, 0, 1, 2, 0, 2, 0, 1, 0, 1, 0, 3, 0, 3, 1, 2, 0, 27, 2, 1, 2, 0],
                     dtype=int).reshape(2, 16),
        ],
        [
            np.array([-10, -10, -10, -9, -9, -8, -8, -8, -8, 2, 2, 3, 3], dtype=np.float64),
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2], dtype=np.int32),
            np.array([-8., 2.], dtype=np.float64),
            np.array([5, 2, 0, 0, 2, 0, 0, 0, 4], dtype=int).reshape(3, 3),
        ]
    ])
    def test_get_initial_tables(self, data_x, data_y, init_splits, expected):
        tm = TableManager(data_x, data_y)
        act_table = tm.create_table(init_splits)
        assert np.all(act_table == expected)

    def test_input_not_sorted_raises_error(self):
        data_x = np.array([-10, -10, -10, -9, -9, -8, -8, -8, -8, 3, 3, 2, 2], dtype=np.float64)
        data_y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2], dtype=np.int32)

        with pytest.raises(ValueError):
            TableManager(data_x, data_y)

    def test_input_not_ndarray_raises_error(self):
        data_x = (np.array([-10, -10, -10, -9, -9, -8, -8, -8, -8, 2, 2, 2, 2], dtype=np.float64),)
        data_y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2], dtype=np.int32)

        with pytest.raises(ValueError):
            TableManager(data_x, data_y)

        data_x = np.array([-10, -10, -10, -9, -9, -8, -8, -8, -8, 2, 2, 2, 2], dtype=np.float64)
        data_y = (np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2], dtype=np.int32),)

        with pytest.raises(ValueError):
            TableManager(data_x, data_y)

    def test_input_incorrect_dtypes_raises_error(self):
        data_x = np.array([-10, -10, -10, -9, -9, -8, -8, -8, -8, 2, 2, 2, 2], dtype=np.float64)
        data_y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2], dtype=np.float64)

        with pytest.raises(ValueError):
            TableManager(data_x, data_y)

    def test_compress_table(self):
        input_table = np.array([2, 1, 0, 0, 2, 4, 0, 3, 0], dtype=int).reshape(3, 3)
        expected_tables = [
            np.array([3, 0, 2, 4, 3, 0], dtype=int).reshape(3, 2),
            np.array([2, 1, 0, 6, 0, 3], dtype=int).reshape(3, 2),
        ]

        fusinter = TableManager(np.array([1, 2, 3]), np.array([1, 1, 1]))
        for i, expected_table in enumerate(expected_tables):
            assert np.all(fusinter.compress_table(input_table, i) == expected_table)
