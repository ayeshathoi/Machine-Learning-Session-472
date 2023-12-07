import numpy as np
import random as r


dimension = input("Enter the dimension of the matrix: ")
dimension = int(dimension)
A = np.random.randint(10, size=(dimension, dimension))
while np.linalg.matrix_rank(A) < dimension:
    M = np.random.randint(10, size=(dimension, dimension))

#Find the eigenvalues and eigenvectors of a random matrix
eigenvalues, eigenvectors = np.linalg.eig(A)

#reconstruct the original matrix
D = np.diag(eigenvalues)
Vector = eigenvectors
Vector_inv = np.linalg.inv(Vector)
A_reconstructed = np.matmul(np.matmul(Vector, D), Vector_inv)

# Round the reconstructed matrix to the nearest integer
A_reconstructed = np.rint(A_reconstructed)
# from complex to Real integer
A_reconstructed = A_reconstructed.astype(int)
# Print the original matrix and the reconstructed matrix
print("The original matrix is: ")
print(A)
print("The reconstructed matrix is: ")
print(A_reconstructed)

# check if the matrix is reconstructed successfully
if np.allclose(A, A_reconstructed) == True:
    print("The matrix is reconstructed successfully")
else:
    print("The matrix is not reconstructed successfully")