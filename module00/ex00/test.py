from matrix import Matrix, Vector


m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.shape)

print(m1.T())
print(m1.T().shape)


m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
             [0.0, 2.0, 4.0, 6.0]])

m2 = Matrix([[0.0, 1.0],
             [2.0, 3.0],
             [4.0, 5.0],
             [6.0, 7.0]])

print(m1 * m2)


m1 = Matrix([[0.0, 1.0, 2.0],
             [0.0, 2.0, 4.0]])

v1 = Vector([[1], [2], [3]])
print(m1 * v1)

v1 = Vector([[1], [2], [3]])
v2 = Vector([[2], [4], [8]])
print(v1 + v2)


