import numpy as np
from scipy.linalg.sqrtm

A = np.array([[1, -2, 3, 5],
              [2, 2, -1, 0],
              [3, 0, 1, 2],
              [1, 0, 2, 0]])
print(np.linalg.matrix_rank(A))
b
B = np.array([
    [1,-2],
    [2,2]
])
sqrtB = np.sqrtm(B)
print(sqrtB)
