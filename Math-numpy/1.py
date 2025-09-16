# Scalars, Vectors, Dot Product, Matrix Multiplication
#-------------------------------------------------------------
# Create two vectors: x = [2, 4, 6], y = [1, 3, 5].
# Compute their sum.
# Compute 3 * x.
# Compute dot product.
# Create two 2×2 matrices:
# M1 = [[2, 0],
#       [1, 3]]
# M2 = [[1, 4],
#       [2, 5]]
# Multiply M1 × M2 using np.matmul.
# (Think) If dot product = 0, what does that mean about the two vectors?

import numpy as np

x = np.array([2, 4, 6])
y = np.array([1, 3, 5])

sum = x + y
print(sum)

scalar_multiplication = 3 * x
print(scalar_multiplication)

M1 = np.array([[2, 0],
               [1, 3]])
M2 = np.array([[1, 4],
               [2, 5]])
matrix_multiplication = np.matmul(M1, M2)
print(matrix_multiplication)

# Slicing, Reshaping, Broadcasting
#-------------------------------------------------------------

# Create a NumPy array nums = [5, 10, 15, 20, 25, 30].
# Slice the first 3 elements.
# Slice every second element.
# Create a NumPy array of numbers 1–12.
# Reshape into a 3×4 matrix.
# Select the second row.
# Select the last column.
# Let:
# A = np.array([[1, 2],
#               [3, 4],
#               [5, 6]])
# b = np.array([10, 20])
# Use broadcasting to add b to each row of A.
# (Think) Why is broadcasting useful in ML when handling datasets with mean normalization? 


# arr[start:stop:step] → 1D slicing.
# arr[row_slice, col_slice] → 2D slicing.
# arr.reshape(new_shape)  new_shape --> row, column  and -1 -> auto calculate any one dimention



nums = np.array([5, 10, 15, 20, 25, 30])
print(nums[:3])
print(nums[::2])

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
arr = np.arange(1,13)
print(arr)
new_shape = arr.reshape(3, 4)
print(new_shape)
print(new_shape[1, :])
print(new_shape[:, -1])

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

b = np.array([10, 20])
broadcasted = A + b
print(broadcasted)


# Compute determinant of:
# M = [[2, 5],
#      [1, 3]]
# Compute inverse of:
# N = [[4, 7],
#      [2, 6]]
# Verify N × N⁻¹ = I.
# Compute rank of:
# P = [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]]
# (Think) Why would a regression model fail if your feature matrix is singular?


M = [[2, 5],
     [1, 3]]
det_M = np.linalg.det(M)
print(det_M)

N = [[4, 7],
     [2, 6]]
inv_N = np.linalg.inv(N)
print(inv_N)
print(np.matmul(N, inv_N))

P = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
det_P = np.linalg.det(P)     
rank_P = np.linalg.matrix_rank(P)
print(det_P)
print(rank_P)