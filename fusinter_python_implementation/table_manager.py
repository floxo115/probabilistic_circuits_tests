import numpy as np


class TableManager:
    """
    TableManager is used to create new Tables and to merge existing ones
    """
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray):

        if not isinstance(data_x, np.ndarray) or not isinstance(data_y, np.ndarray):
            raise ValueError("data inputs have to be of type ndarray")
        if not np.issubdtype(data_y.dtype, np.integer):
            raise ValueError("data_x has to be of type float and data_y has to be of type int")
        if not TableManager._is_sorted(data_x):
            raise ValueError("data has to be sorted in ascending order")

        self.data_x = data_x
        self.data_y = data_y

    def create_table(self, init_splits: np.ndarray) -> np.ndarray:
        """
        creates a table from the initial splits of the data
        :return: np.ndarray with k columns and n rows from k splits and n classes
        """

        # get the dimensions of the new table
        n_labels = len(np.unique(self.data_y))
        n_splits = len(init_splits) + 1

        # create zeroed table with the needed shape
        table = np.zeros((n_labels, n_splits), dtype=int)

        # create column for accumulating the entries in the interval
        n_labels_in_interval = np.zeros(n_labels, dtype=int)
        i = 0

        # go over dataset until a interval split is reached
        # accumulate all seen values in a column vector and
        # then set the according column of the table to have
        # the values of that vector
        for split_idx, split_val in enumerate(init_splits):
            while self.data_x[i] < split_val:
                n_labels_in_interval[self.data_y[i]] += 1
                i += 1

            table[:, split_idx] = n_labels_in_interval
            n_labels_in_interval[:] = 0

        # for all remaining examples accumulate them and put
        # the resulting vector in the last column of the table
        while i < len(self.data_x):
            n_labels_in_interval[self.data_y[i]] += 1
            i += 1
        table[:, n_splits - 1] = n_labels_in_interval

        return table
    @staticmethod
    def compress_table(input_table: np.ndarray, i: int) -> np.ndarray:
        """
        returns a table like the one returned by create_table but with the i-th and (i+1)-th columns merged
        :param input_table: a table like the one returned by create_table
        :param i: the index for the merge
        :return: a table like the input but with 2 consecutive columns merged
        """
        n, m = input_table.shape

        if i < 0 or (m - 1) < i:
            raise ValueError(f"the parameter i has to have values between 0 and (len of columns -1), but is {i}")

        new_table = np.zeros((n, m - 1), dtype=int)

        # We iterate over the new and the old table to compute the values for the new table
        init_table_index, new_table_index = 0, 0
        while init_table_index < m:
            if init_table_index == i:
                # if we are in the first column of the merged columns we add their values together and insert the result
                # into the new table
                new_table[:, new_table_index] = np.sum(input_table[:, init_table_index:(init_table_index + 2)], axis=1)
                new_table_index += 1
            elif init_table_index != i + 1:
                # if we are in no column to be merged we simply copy the values to the new table
                new_table[:, new_table_index] = input_table[:, init_table_index]
                new_table_index += 1

            # the case of being in the second merge column has no actions except not incrementing the new table index
            # in any case the init table index has to be incremented
            init_table_index += 1

        return new_table

    @staticmethod
    def _is_sorted(x):
        return np.all(x[:-1] <= x[1:])
