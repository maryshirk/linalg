import unittest
from main import Matrix, calculate_sparse_determinant


class Task3Test(unittest.TestCase):
    def test_3_x_3_matrix(self):
        # Тестирование для матрицы 3x3
        matrix_a = Matrix(3, 3)
        matrix_a.data = [1.0, 2.0, 1.0, 2.0, 8.0, 7.0, 1.0, 4.0]
        matrix_a.ind = [0, 1, 2, 0, 1, 2, 0, 2]
        matrix_a.indptr = [0, 3, 6, 8]

        determinant_result = calculate_sparse_determinant(matrix_a)
        expected_result = 22
        self.assertEqual(determinant_result, expected_result)

    def test_float_matrix(self):
        # Тестирование для матрицы с вещественными числами
        matrix_b = Matrix(4, 4)
        matrix_b.data = [0.5, 2.0, 9.0, 8.0, 1.0, 9.5, 9.5, 3.0, 2.5, 6.0, 8.0, 1.0, 3.0, 8.0, 2.5, 1.0]
        matrix_b.ind = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        matrix_b.indptr = [0, 4, 8, 12, 16]

        determinant_result = calculate_sparse_determinant(matrix_b)
        expected_result = -982.125
        self.assertAlmostEqual(determinant_result, expected_result)

    def test_4_x_4_matrix(self):
        # Тестирование для матрицы 4x4
        matrix_c = Matrix(4, 4)
        matrix_c.data = [1.0, 2.0, -1.0, 2.0, 2.0, 3.0, -1.0, 1.0, 2.0, 1.0, 8.0, 1.0, 3.0, 6.0, 2.0]

        matrix_c.ind = [0, 1, 2, 3, 0, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3]

        matrix_c.indptr = [0, 4, 7, 11, 15]

        determinant_result = calculate_sparse_determinant(matrix_c)
        expected_result = 64
        self.assertEqual(determinant_result, expected_result)

    def test_6_x_6_matrix(self):
        # Тестирование для матрицы 6x6
        matrix_d = Matrix(6, 6)
        matrix_d.data = [1.0, 2.0, 9.0, 12.0, 3.0, 4.0, 8.0, 7.0, 3.0, -9.0, 2.0, 1.0, 6.0, 6.0, 6.0, 3.0, 2.0, -12.0,
                         -8.0, 6.0, -12.0, 5.0, 48.0, 2.0, 5.0, 7.0, 1.0, 11.0, 9.0, 43.0, 1.0, -32.0, 6.0, 4.0, 3.0,
                         1.0]

        matrix_d.ind = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1,
                        2, 3, 4, 5]

        matrix_d.indptr = [0, 6, 12, 18, 24, 30, 36]

        determinant_result = calculate_sparse_determinant(matrix_d)
        expected_result = 50369304
        self.assertEqual(determinant_result, expected_result)

    def test_null_det(self):
        # Тестирование для матрицы с определителем равным нулю
        matrix_e = Matrix(3, 3)
        matrix_e.data = [1.0, 2.0, -98.0, 1.0, 7.0, 2.0]
        matrix_e.ind = [1, 2, 1, 2, 1, 2]
        matrix_e.indptr = [0, 2, 4, 6]

        determinant_result = calculate_sparse_determinant(matrix_e)
        expected_result = -0
        self.assertEqual(determinant_result, expected_result)


if __name__ == '__main__':
    unittest.main()
