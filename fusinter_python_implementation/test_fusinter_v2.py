import numpy as np

from datasets.paper_dataset import paper_dataset_x, paper_dataset_y
from fusinter_v1 import FUSINTERDiscretizer as FUSINTERDiscretizer_v1
from .fusinter_v2 import FUSINTERDiscretizer


class TestFusinterV2:
    def test_if_apply_give_same_result_as_v1(self):
        for alpha in np.linspace(0.01, 1, 10):
            for lam in np.linspace(0.01, 2, 20):
                v1 = FUSINTERDiscretizer_v1(paper_dataset_x, paper_dataset_y)
                v1_splits = v1.apply(alpha, lam)
                v2 = FUSINTERDiscretizer(alpha, lam)
                v2_splits = v2.fit(paper_dataset_x, paper_dataset_y)


                if len(v1_splits) == 0:
                    assert len(v2_splits) == 0
                else:
                    assert np.all(v1_splits == v2_splits)
