import unittest
from main import Matrix, matrix_addition, multiply_by_scalar, matrix_multiplication


class Task2Test(unittest.TestCase):
    def setUp(self):
        # Подготовка тестовых данных
        self.matrix_a = Matrix(2, 2)
        self.matrix_a.data = [1.0, 2.0, 3.0]
        self.matrix_a.ind = [0, 1, 1]
        self.matrix_a.indptr = [0, 2, 3]

        self.matrix_b = Matrix(2, 2)
        self.matrix_b.data = [4.0, 5.0, 6.0, 7.0]
        self.matrix_b.ind = [0, 1, 0, 1]
        self.matrix_b.indptr = [0, 2, 4]

        self.empty_matrix = Matrix(0, 0)

    def test_addition(self):
        # Сложение матриц
        result_matrix = matrix_addition(self.matrix_a, self.matrix_b)
        expected_result = Matrix(2, 2)
        expected_result.data = [5.0, 7.0, 6.0, 10.0]
        expected_result.ind = [0, 1, 0, 1]
        expected_result.indptr = [0, 2, 4]

        self.assertEqual(result_matrix.data, expected_result.data)
        self.assertEqual(result_matrix.indptr, expected_result.indptr)

    def test_addition_different_sizes(self):
        # Сложение матриц несовпадающих размеров
        matrix_c = Matrix(2, 3)
        matrix_c.data = [1.0, 2.0, 3.0]
        matrix_c.ind = [0, 1]
        matrix_c.indptr = [0, 2]

        with self.assertRaises(ValueError):
            matrix_addition(self.matrix_a, matrix_c)

    def test_addition_empty_matrix(self):
        # Сложение пустых матриц
        result_matrix = matrix_addition(self.empty_matrix, self.empty_matrix)

        self.assertEqual(result_matrix.n, 0)
        self.assertEqual(result_matrix.m, 0)

    def test_scalar_multiplication(self):
        # Умножение матрицы на скаляр
        scalar = 3
        result_matrix = multiply_by_scalar(self.matrix_a, scalar)

        expected_result = Matrix(2, 2)
        expected_result.data = [3.0, 6.0, 9.0]
        expected_result.ind = [0, 1, 1]
        expected_result.indptr = [0, 2, 3]

        self.assertEqual(result_matrix.data, expected_result.data)
        self.assertEqual(result_matrix.indptr, expected_result.indptr)

    def test_scalar_multiplication_empty(self):
        # Умножение пустой матрицы на скаляр
        scalar = 5
        result_matrix = multiply_by_scalar(self.empty_matrix, scalar)

        self.assertEqual(result_matrix.n, 0)
        self.assertEqual(result_matrix.m, 0)

    def test_multiplication(self):
        # Умножение матриц
        matrix_c = Matrix(2, 3)
        matrix_c.data = [1.0, 2.0]
        matrix_c.ind = [0, 1]
        matrix_c.indptr = [0, 1, 2]

        matrix_d = Matrix(3, 2)
        matrix_d.data = [4.0, 5.0, 6.0]
        matrix_d.ind = [1, 0, 1]
        matrix_d.indptr = [0, 1, 2, 3]

        result_matrix = matrix_multiplication(matrix_c, matrix_d)

        expected_result = Matrix(2, 2)
        expected_result.data = [4.0, 10.0]
        expected_result.ind = [1, 0]
        expected_result.indptr = [0, 1, 2]

        self.assertEqual(result_matrix.data, expected_result.data)
        self.assertEqual(result_matrix.indptr, expected_result.indptr)

    def test_multiplication_different_sizes(self):
        # Умножение матриц с несовпадающими размерами
        matrix_e = Matrix(4, 2)
        matrix_e.data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        matrix_e.ind = [0, 1, 0, 1, 0, 1, 0, 1]
        matrix_e.indptr = [0, 2, 4, 6, 8]

        with self.assertRaises(ValueError):
            result_matrix = matrix_multiplication(self.matrix_a, matrix_e)

    def test_multiplication_empty(self):
        # Умножение пустых матриц
        result_matrix = matrix_multiplication(self.empty_matrix, self.empty_matrix)

        self.assertEqual(result_matrix.n, 0)
        self.assertEqual(result_matrix.m, 0)


if __name__ == '__main__':
    unittest.main()
