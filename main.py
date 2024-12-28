class Matrix:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.data = []
        self.ind = []
        self.indptr = [0] * (n + 1)

    def from_csr(self):
        rows = self.n
        cols = self.m
        matrix = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(self.indptr[i], self.indptr[i + 1]):
                matrix[i][self.ind[j]] = self.data[j]

        return matrix

    def print_dense_matrix(self):
        dense = self.from_csr()
        for i in dense:
            print(*i)

    def input_matrix(self):
        for i in range(self.n):
            input_row = [float(x) for x in input().split()]
            for j in range(self.m):
                if input_row[j] != 0:
                    self.data.append(input_row[j])
                    self.ind.append(j)
            self.indptr[i + 1] = len(self.data)

    def add_value(self, row, col, value):
        if value != 0:
            self.data.append(value)
            self.ind.append(col - 1)
            for i in range(row, self.n + 1):
                self.indptr[i] += 1

    def get_value(self, row, col):
        if (row < 1 or row > self.n) or (col < 1 or col > self.m):
            raise IndexError('Index out of range')
        row, col = row - 1, col - 1
        start = self.indptr[row]
        end = self.indptr[row + 1]
        for i in range(start, end):
            if self.ind[i] == col:
                return self.data[i]
        return 0.0

    def print_get_value(self, row, col):
        print(self.get_value(row, col))

    def trace(self):
        trace_sum = 0
        if self.n != self.m:
            raise ValueError("Matrix is no square")
        for i in range(self.n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for j in range(start, end):
                if self.ind[j] == i:
                    trace_sum += self.data[j]
                    break
        return trace_sum

    def print_trace(self):
        print(self.trace())

    def print_sparse_matrix(self):
        print(f"Значения: {self.data}")
        print(f"Столбцы: {self.ind}")
        print(f"Указатели: {self.indptr}")


def matrix_addition(matrix_1, matrix_2):
    if matrix_1.n != matrix_2.n or matrix_1.m != matrix_2.m:
        raise ValueError("Sizes is not correct")
    result = Matrix(matrix_1.n, matrix_1.m)
    for row in range(matrix_1.n):
        current_row_sum = {}
        start_1 = matrix_1.indptr[row]
        end_1 = matrix_1.indptr[row + 1]
        for index in range(start_1, end_1):
            col = matrix_1.ind[index]
            value = matrix_1.data[index]
            current_row_sum[col] = value
        start_2 = matrix_2.indptr[row]
        end_2 = matrix_2.indptr[row + 1]
        for index in range(start_2, end_2):
            col = matrix_2.ind[index]
            value = matrix_2.data[index]
            current_row_sum[col] = current_row_sum.get(col, 0) + value
        sorted_columns = sorted(current_row_sum.items(), key=lambda x: x[0])
        for col, total_value in sorted_columns:
            if total_value != 0:
                result.add_value(row + 1, col + 1, total_value)
    return result


def print_matrix_addition(matrix_1, matrix_2):
    result = matrix_addition(matrix_1, matrix_2)
    result.print_sparse_matrix()


def multiply_by_scalar(matrix, scalar):
    result = Matrix(matrix.n, matrix.m)
    if scalar == 0.0:
        result.ind = []
        result.data = []
        result.indptr = []
    else:
        for row in range(matrix.n):
            start = matrix.indptr[row]
            end = matrix.indptr[row + 1]
            for index in range(start, end):
                col = matrix.ind[index]
                value = matrix.data[index]
                result.add_value(row + 1, col + 1, value * scalar)
    return result


def print_multiply_by_scalar(matrix_1, matrix_2):
    result = multiply_by_scalar(matrix_1, matrix_2)
    result.print_sparse_matrix()


def matrix_multiplication(matrix_1, matrix_2):
    if matrix_1.m != matrix_2.n:
        raise ValueError("Sizes are not correct")
    result = Matrix(matrix_1.n, matrix_2.m)
    for i in range(matrix_1.n):
        for j in range(matrix_2.m):
            local_sum = 0
            for index in range(matrix_1.indptr[i], matrix_1.indptr[i + 1]):
                k_col = matrix_1.ind[index]
                a_value = matrix_1.data[index]
                b_value = 0
                for k_index in range(matrix_2.indptr[k_col], matrix_2.indptr[k_col + 1]):
                    if matrix_2.ind[k_index] == j:
                        b_value = matrix_2.data[k_index]
                        break
                local_sum += a_value * b_value
            if local_sum != 0:
                result.add_value(i + 1, j + 1, local_sum)
    return result


def print_matrix_multiplication(matrix_1, matrix_2):
    result = matrix_multiplication(matrix_1, matrix_2)
    result.print_sparse_matrix()


def calculate_sparse_determinant(sparse_matrix):
    if sparse_matrix.n == 1:
        return sparse_matrix.get_value(1, 1)
    elif sparse_matrix.n == 2:
        m00 = sparse_matrix.get_value(1, 1)
        m01 = sparse_matrix.get_value(1, 2)
        m10 = sparse_matrix.get_value(2, 1)
        m11 = sparse_matrix.get_value(2, 2)
        return m00 * m11 - m01 * m10

    det_total = 0
    for col in range(1, sparse_matrix.m + 1):
        main_value = sparse_matrix.get_value(1, col)
        if main_value == 0:
            continue

        minor_matrix = Matrix(sparse_matrix.n - 1, sparse_matrix.m - 1)
        for i in range(2, sparse_matrix.n + 1):
            for j in range(1, sparse_matrix.m + 1):
                if j == col:
                    continue
                minor_matrix.add_value(i - 1, j if j < col else j - 1, sparse_matrix.get_value(i, j))

        current_minor_det = calculate_sparse_determinant(minor_matrix)
        sign = (-1) ** (col + 1)
        det_total += sign * main_value * current_minor_det

    return det_total


def determinant_and_invertibility(sparse_version):
    determinant = calculate_sparse_determinant(sparse_version)
    is_invertible = determinant != 0

    print(f'Определитель матрицы: {determinant}')
    print(f'Существует ли матрица обратная данной {is_invertible}')
    if is_invertible:
        print("да")
    else:
        print("нет")


if __name__ == "__main__":
    while True:
        n = int(input('Введите количество строк: '))
        m = int(input('Введите количество столбцов: '))
        matrix = Matrix(n, m)
        matrix.input_matrix()


        print('Скажите, что Вы хотите найти')
        print('1 - Перевести матрицу в CSR формат')
        print('2 - След матрицы')
        print('3 - Запрос элемента матрицы по индексам')
        print('4 - Сложение матриц')
        print('5 - Умножение матрицы на скаляр')
        print('6 - Умножение матриц')
        print('7 - Определитель матрицы')
        print('0 - Выход')



        choice = input('Введите номер операции: ')

        if choice == '0':
            break

        if choice == '1':
            matrix.print_sparse_matrix()

        elif choice == '2':
            matrix.print_trace()

        elif choice == '3':
            row = int(input('Введите номер строки (индексация с 1): '))
            col = int(input('Введите номер столбца (индексация с 1): '))
            matrix.print_get_value(row, col)

        elif choice == '4':
            n_other = int(input('Введите количество строк второй матрицы: '))
            m_other = int(input('Введите количество столбцов второй матрицы: '))
            matrix_other = Matrix(n_other, m_other)
            matrix_other.input_matrix()
            matrix_addition(matrix, matrix_other).print_dense_matrix()

        elif choice == '5':
            scalar = float(input('Введите скаляр: '))
            result_scalar_multiply = multiply_by_scalar(matrix, scalar)
            print(f'Результат умножения на скаляр {scalar}:')
            # result_scalar_multiply.print_sparse_matrix()
            result_scalar_multiply.print_dense_matrix()

        elif choice == '6':
            n_other = int(input('Введите количество строк второй матрицы: '))
            m_other = int(input('Введите количество столбцов второй матрицы: '))
            matrix_other = Matrix(n_other, m_other)
            matrix_other.input_matrix()
            matrix_multiplication(matrix, matrix_other).print_dense_matrix()

        elif choice == '7':
            determinant_and_invertibility(matrix)

        else:
            print('Некорректный ввод. Пожалуйста, выберите номер операции от 0 до 7.')
