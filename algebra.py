import time
import numpy as np
import matplotlib.pyplot as plt

class Matrix:
    @staticmethod
    def random_matrix(n):
        return [[np.random.randint(0, 10) for _ in range(n)] for _ in range(n)]

    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __str__(self):
        return '\n'.join(['\t'.join(map(str,row)) for row in self.data])
    
    def add(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices should be the same size!")
        return Matrix([[self.data[i][j] + other.data[i][j]
                        for j in range(self.cols)]
                        for i in range(self.rows)])
    
    def substract(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices should be the same size!")
        return Matrix([[self.data[i][j] - other.data[i][j]
                        for j in range(self.cols)]
                        for i in range(self.rows)])
    
    def multiply(self, other):
        if self.cols != other.rows:
            raise ValueError("Columns of A must match rows of B")
        result = [[sum(self.data[i][k] * other.data[k][j] 
                       for k in range(self.cols))
                       for j in range(other.cols)]
                       for i in range(self.rows)]
        return Matrix(result)
    
    @staticmethod
    def determinant(matrix):
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        if n == 2:
            return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
        
        det = 0
        for c in range(n):
            sub_matrix = [row[:c] + row[c+1:] for row in matrix[1:]]
            det += ((-1)**c) * matrix[0][c] * Matrix.determinant(sub_matrix)
        return det

    @staticmethod
    def inverse(matrix):
        n = len(matrix)
        det = Matrix.determinant(matrix)
        if det == 0:
            raise ValueError("Singular matrix, not possible to invert.")
        
        cofactors = []
        for r in range(n):
            cofactor_row = []
            for c in range(n):
                minor = [row[:c] + row[c+1:]
                         for i,row in enumerate(matrix) if i != r]
                cofactor_row.append(
                    ((-1)**(r+c)) * Matrix.determinant(minor)
                )
            cofactors.append(cofactor_row)

        adjoint = [[cofactors[j][i] for j in range(n)] for i in range(n)]
        inverse_matrix = [
            [adjoint[i][j]/det for j in range(n)]
            for i in range(n)
        ]
        return inverse_matrix
    
    @staticmethod
    def gaussian_solve(A, b):
        n = len(A)
        for i in range(n):
            max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]

            for j in range(i+1, n):
                ratio = A[j][i]/A[i][i]
                for k in range(i, n):
                    A[j][k] -= ratio*A[i][k]
                b[j] -= ratio*b[i]

        x = [0]*n
        for i in range(n-1, -1, -1):
            x[i] = (b[i] - sum(A[i][j]*x[j]
                               for j in range(i+1, n))) / A[i][i]
        return x


if __name__ == "__main__":
    M1 = Matrix([[1,2],[3,4]])
    M2 = Matrix([[5,6], [7,8]])

    print("A + B:\n", M1.add(M2))
    print("A - B:\n", M1.substract(M2))
    print("A * B:\n", M1.multiply(M2))

    mat = [[4, 7],[2, 6]]
    print("Determinant of mat:", Matrix.determinant(mat))
    print("Inverse of mat:", Matrix.inverse(mat))

    A_sys = [[2, 1, -1], [-3, -1, 2], [-2,1,2]]
    b_sys = [8, 11, -3]

    print("Solution to AX = b:",
          Matrix.gaussian_solve([row[:] for row in A_sys], b_sys[:]))

    A_np = np.array(A_sys)
    b_np = np.array(b_sys)
    start = time.time()
    x_np = np.linalg.solve(A_np, b_np)
    end = time.time()
    print("NumPy solution:", x_np, "Time:", end-start)

    sizes = [10, 20, 40, 60, 80, 100]
    custom_times = []
    numpy_times = []

    for n in sizes:
        A = Matrix.random_matrix(n)
        b = [np.random.randint(0, 10) for _ in range(n)]

        A_copy = [row[:] for row in A]
        b_copy = b[:]
        start = time.time()
        Matrix.gaussian_solve(A_copy, b_copy)
        custom_times.append(time.time() - start)

        A_np = np.array(A)
        b_np = np.array(b)
        start = time.time()
        np.linalg.solve(A_np, b_np)
        numpy_times.append(time.time() - start)

sizes = [10, 50, 100, 200, 400, 600]
custom_times = []
numpy_times = []

for n in sizes:
    A = Matrix.random_matrix(n)
    b = np.random.randint(0, 10, size=n).astype(float).tolist()

    A_copy = [row[:] for row in A]
    b_copy = b[:]
    start = time.perf_counter()
    Matrix.gaussian_solve(A_copy, b_copy)
    custom_times.append(time.perf_counter() - start)

    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    t_np = []
    for _ in range(5):
        start = time.perf_counter()
        np.linalg.solve(A_np, b_np)
        t_np.append(time.perf_counter() - start)
    numpy_times.append(sum(t_np) / 5)

plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(10, 6), dpi=150)

plt.plot(sizes, custom_times, marker='o', linewidth=2.0,
         markersize=7, label="Custom Gaussian Elimination", color="#1f77b4")

plt.plot(sizes, numpy_times, marker='x', linewidth=2.0,
         markersize=7, label="NumPy", color="#ff7f0e")

plt.yscale("log")
plt.title("Performance Comparison: Custom Gaussian Elimination vs NumPy", fontsize=16)
plt.xlabel("Matrix Size (n × n)", fontsize=13)
plt.ylabel("Computation Time (seconds, log scale)", fontsize=13)
plt.xticks(sizes, fontsize=11)
plt.yticks(fontsize=11)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("benchmark_plot.png", dpi=150)
plt.show()