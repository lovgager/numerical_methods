import numpy as np


class Sparse:
    """
    A class representing a sparse matrix.

    Attributes:
        rows (np.ndarray): An array containing row indices of non-zero elements.
        cols (np.ndarray): An array containing column indices of non-zero elements.
        values (np.ndarray): An array containing values of non-zero elements.
    """

    def __init__(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray):
        self.rows = np.array(rows, dtype=int)
        self.cols = np.array(cols, dtype=int)
        self.values = values

    def multiply(self, vector: np.ndarray) -> np.ndarray:
        """
        Performs matrix-vector multiplication with a given vector.

        Args:
            vector (np.ndarray): The vector to be multiplied with the sparse matrix.

        Returns:
            np.ndarray: The result of the multiplication.
        """
        one_row_multiply = lambda r: vector[self.cols[self.rows == r]] \
            @ self.values[self.rows == r]
        m = np.max(self.rows) + 1
        return np.vectorize(one_row_multiply)(np.arange(m))

    def transpose(self):
        """
        Returns the transpose of the sparse matrix.

        Returns:
            Sparse: The transpose of the sparse matrix.
        """
        return Sparse(self.cols, self.rows, self.values)

    def get_row(self, row_index: int, col_slice: tuple[int]) -> np.ndarray:
        """
        Retrieves a specific row from the sparse matrix within the specified column slice.

        Args:
            row_index (int): The index of the row to retrieve.
            col_slice (tuple[int]): A tuple representing the start and end indices of the column slice.

        Returns:
            np.ndarray: The requested row.
        """
        start = col_slice[0]
        end = col_slice[1]
        if start >= end:
            return np.zeros(0)
        res = np.zeros(end - start)
        cols = np.array(self.cols, dtype=int)
        rows = np.array(self.rows, dtype=int)
        values_row = self.values[self.rows == row_index]
        indices = cols[rows == row_index]
        values_row = values_row[start <= indices]
        indices = indices[start <= indices]
        values_row = values_row[indices < end]
        indices = indices[indices < end]
        if len(indices) > 0:
            res[indices - start] = values_row
        return res
        

    def get_element(self, row_index: int, col_index: int):
        """
        Retrieves the value of a specific element in the sparse matrix.

        Args:
            row_index (int): The row index of the element.
            col_index (int): The column index of the element.

        Returns:
            float: The value of the specified element.
        """
        res = self.values[(self.rows == row_index)*(self.cols == col_index)]
        if len(res) > 0:
            return res[0]
        return 0

    def to_dense(self):
        """
        Converts the sparse matrix to a dense matrix.

        Returns:
            np.ndarray: The dense matrix representation of the sparse matrix.
        """
        m = np.max(self.rows) + 1
        n = np.max(self.cols) + 1
        dense = np.zeros((m, n))
        dense[self.rows, self.cols] = self.values
        return dense


#  для самопроверки

rows = np.array([0, 1, 2, 2])
cols = np.array([0, 1, 1, 2])
values = np.array([3, 4, 5, 6])
sparse_matrix = Sparse(rows, cols, values)

# Умножение матрицы на вектор
vector = np.array([1, 2, 3])
result = sparse_matrix.multiply(vector)
assert vector.size == result.size
assert np.max(np.absolute(result - np.array([3, 8, 28]))) < 1.0e-30


# Получить строку по индексу строки и диапазону столбцов
row_index = 2
column_start = 0
column_end = 3
row = sparse_matrix.get_row(row_index, (column_start, column_end))
assert row.size == column_end - column_start

# Получить элемент по индексу строки и столбца
element = sparse_matrix.get_element(2, 1)
assert abs(element - 5) < 1.0e-30

assert np.max(np.absolute(sparse_matrix.to_dense() - np.array([[3., 0., 0.],[0., 4., 0.],[0., 5., 6.]]))) < 1.0e-30
