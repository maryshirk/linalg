import unittest
from main import Matrix


class Task1Test(unittest.TestCase):
    def setUp(self):
        # Пример разреженной матрицы 3x3
        self.matrix_data = [0, 0, 0,
                            5, 0, 0,
                            0, 3, 0]
        self.sparse_matrix = Matrix(3, 3)

        # Инициализация матрицы в формате CSR
        self.sparse_matrix.data = [5.0, 3.0]
        self.sparse_matrix.ind = [0, 1]
        self.sparse_matrix.indptr = [0, 0, 1, 2]

    def test_trace_null_square_matrix(self):
        # Тестирование следа нулевой квадратной матрицы
        null_matrix = Matrix(3, 3)
        null_matrix.data = []
        null_matrix.ind = []
        null_matrix.indptr = [0, 0, 0, 0]

        self.assertEqual(null_matrix.trace(), 0)

    def test_trace_non_square_matrix(self):
        # Тестирование следа не квадратной матрицы
        non_square_matrix = Matrix(3, 2)
        non_square_matrix.data = [1.0, 2.0, 3.0, 4.0]
        non_square_matrix.ind = [0, 1, 0, 1]
        non_square_matrix.indptr = [0, 2, 4, 4]

        with self.assertRaises(ValueError):
            non_square_matrix.trace()

    def test_get_element_existing(self):
        # Тест на получение ненулевого элемента
        self.assertEqual(self.sparse_matrix.get_value(2, 1), 5.0)

    def test_get_element_non_existing(self):
        # Тест на получение нулевого элемента
        self.assertEqual(self.sparse_matrix.get_value(1, 2), 0.0)

    def test_get_element_out_of_bounds(self):
        # Тест на выход за границы
        with self.assertRaises(IndexError):
            self.sparse_matrix.get_value(4, 1)

        with self.assertRaises(IndexError):
            self.sparse_matrix.get_value(1, 4)

    def test_empty_matrix(self):
        # Тест на пустую матрицу
        empty_matrix = Matrix(0, 0)

        self.assertEqual(empty_matrix.from_csr(), [])
        self.assertEqual(empty_matrix.trace(), 0)


if __name__ == "__main__":
    unittest.main()
