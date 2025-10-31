import numpy as np

from .splitvaluecomputer import SplitValueComputer
from .splitter import Splitter
from .table_manager import TableManager
from time import perf_counter

class FUSINTERDiscretizer:
    """
    FUSINTERDiscretizer takes a numpy array of values and one of the according labels and discretize the values according
    to the FUSINTER algorithm.
    """

    def __init__(self, alpha=0.95, lam=1, min_examples_in_interval=-1):
        """
        Initialized the FUSINTER discretizer
        :param alpha: a float value for parametrizing the split value estimation
        :param lam:  a float value for parametrizing the split value estimation
        """
        self.alpha = alpha
        self.lam = lam
        self.data_x = np.array([])
        self.data_x = np.array([])
        self.computed_splits = np.array([])
        self.min_examples_in_interval = min_examples_in_interval

    def _init_dataset(self,
                      data_x: np.ndarray,
                      data_y: np.ndarray,
                      ):
        """
        Initializes the given data for processing
        :param data_x: a 1D array of values
        :param data_y: a 1D array of labels
        """
        if not isinstance(data_x, np.ndarray):
            raise ValueError("data_x should be a numpy array")
        if data_x.ndim != 1:
            raise ValueError(f"the number of dims of data_x should be 1 {data_y.shape}")

        if not isinstance(data_y, np.ndarray):
            raise ValueError("data_y should be a numpy array")
        if data_y.ndim != 1:
            raise ValueError(f"the number of dims of data_y should be 1 {data_y.shape}")
        if not np.issubdtype(data_y.dtype, np.integer):
            raise ValueError(f"the dtype of data_y should be int32, but is {data_y.dtype}")

        if len(data_x) != len(data_y):
            raise ValueError("the given arrays should be of the same size, but are not")

        self.data_x = data_x.copy()
        self.data_y = data_y.copy()

        # For getting labels in the range 0..k
        label_masks = []
        for label in np.unique(self.data_y):
            label_masks.append(self.data_y == label)

        for i, label_mask in enumerate(label_masks):
            self.data_y[label_mask] = i

        # sort the given dataset
        sort_idx = np.argsort(self.data_x)
        self.data_y = self.data_y[sort_idx]
        self.data_x = self.data_x[sort_idx]

        # get necessary instances for applying FUSINTER
        self.splitter = Splitter(self.data_x, self.data_y, self.min_examples_in_interval)
        self.table_manager = TableManager(self.data_x, self.data_y)

    def transform(self, data):
        """
        Makes given data discrete after fitting the instance
        :param data: numpy array of data
        :return: numpy array of discrete data. If instance is not fitted: returns None
        """
        if self.computed_splits is not None:
            return np.digitize(data, self.computed_splits)
        else:
            return None

    def fit(self, data_x: np.ndarray, data_y: np.ndarray) -> np.ndarray:
        """
        :alpha: alpha float parameter for the entropy merge criterion
        :lam: lambda float parameter for the entropy merge criterion
        :return: a numpy array of continuous interval split points for the discretization of the classes dataset
        """

        # get init splits and init table
        self._init_dataset(data_x, data_y)
        splits, _ = self.get_initial_intervals()
        table = self.create_table(splits)

        # instantiate the MergeValueComputer and run FUSINTER until no more splits can be removed
        # start = perf_counter()
        svc = SplitValueComputer(table, splits, self.alpha, self.lam)
        # end = perf_counter()
        # print("init time", end - start)
        while len(svc.heap) >= 1:
            max_value = svc.get_max_delta()
            #print("heap size", len(svc.heap), flush=True)
            # print(svc.heap.array[0].sort_values[1], max_value)

            if max_value <= 0:
                break

            svc.merge_max()

        return svc.get_splits()

    def get_initial_intervals(self):
        """
        return  the initial intervals computed by the Splitter
        """
        return self.splitter.apply()

    def create_table(self, init_splits: np.ndarray) -> np.ndarray:
        """
        creates a table from the initial splits of the data
        :return: np.matrix with k columns and n rows from k splits and n classes
        """
        return self.table_manager.create_table(init_splits)

    def compress_table(self, input_table: np.ndarray, i: int) -> np.ndarray:
        """
        returns a table like the one returned by create_table but with the i-th and (i+1)-th columns merged
        :param input_table: a table like the one returned by create_table
        :param i: the index for the merge
        :return: a table like the input but with 2 consecutive columns merged
        """
        return self.table_manager.compress_table(input_table, i)
