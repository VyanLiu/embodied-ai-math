print("environment_setting_success")
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A - B =\n", A - B)
print("A * B =\n", A * B)
print("A @ B =\n", A @ B)
print("A Transform = \n", A.T)
print("A Inverse = \n", np.linalg.inv(A))
print("A Delta = \n", np.linalg.det(A))

print("let's get start tomorrow!\n")