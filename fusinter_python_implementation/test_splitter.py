import numpy as np
import pytest

from datasets import paper_dataset
from .splitter import Splitter


class TestSplitter:

    @pytest.mark.parametrize("data_x,data_y,exp_splits", [
        (
                paper_dataset.paper_dataset_x,
                paper_dataset.paper_dataset_y,
                np.array([2, 3, 13, 14, 15, 16, 17, 18, 19, 20, 23, 37, 38, 39, 40], dtype=np.float64),
        ),
        (
                np.array([-10, -10, -10, -9, -9, -8, -8, -8, -8, 2, 2, 3, 3], dtype=np.float64),
                np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2], dtype=np.int32),
                np.array([-8., 2.], dtype=np.float64),
        )
    ])
    def test_get_initial_intervals(self, data_x, data_y, exp_splits):
        splitter = Splitter(data_x, data_y)
        act_splits, act_labels = splitter.apply()
        assert np.all(act_splits == exp_splits)
        # assert np.all(act_labels == exp_labels)
