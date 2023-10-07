from numpy import *
import matplotlib.pyplot as plt

#M = 10
M = int(input("M = "))
h = (2 - 1)/M

A = [[0 for i in range(M - 1)] for j in range(M - 1)]

for i in range(M - 2):
    A[i][i] = 2/h
    A[i + 1][i] = -1/h
    A[i][i + 1] = -1/h
A[M - 2][M - 2] = 2/h


x = [1 + i * h for i in range(1, M)]
f = [h * x[i] for i in range(M - 1)]


invA = linalg.inv(A)
u = matmul(invA, f).tolist()

x = [1] + x + [2]
u = [0] + u + [0]

#print("A =")
#for i in A:
#    print(i)

#print("x =", x)
#print("f =", f)
#print("u =", u)

x_egz = arange(1.0, 2.0, 0.00001)
u_egz = -1/6 * x_egz**3 + 7/6 * x_egz - 1


plt.plot(x, u)
plt.plot(x_egz, u_egz)

plt.xlabel('x - os')
plt.ylabel('y - os')
plt.title("Funkcija i njena FEM-aproksimacija s M = " + str(M))

plt.show()