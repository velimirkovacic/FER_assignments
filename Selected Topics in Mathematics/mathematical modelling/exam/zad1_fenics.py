import fenics as fn
import matplotlib.pyplot as plt
import numpy as np

#M = 10
M = int(input("M = "))

#   mesh = diskretizacija domene
#   V = prostor konacnih elemenata
mesh = fn.IntervalMesh(M, 1, 2)
V = fn.FunctionSpace(mesh, 'CG', 1) #u 1D

#   u_D = Dirichletov rubni uvjet, u(1) = u(2) = 0
u_D = fn.Expression('0', degree=1)


def boundary(x, on_boundary):
	return on_boundary

bc = fn.DirichletBC(V, u_D, boundary)


# Slaba formulacija, bazne i testne funkcije
u = fn.TrialFunction(V)
v = fn.TestFunction(V)
f = fn.Expression('x[0]', degree=1)
a = fn.dot(fn.grad(u), fn.grad(v))*fn.dx
L = f*v*fn.dx

# Kreiranje matrice sustava i vektora desne strane
A = fn.assemble(a)
b = fn.assemble(L)

bc.apply(A, b)

# Rjesavanje sustava linearnih jednadzbi
u = fn.Function(V)
U = u.vector()
fn.solve(A, U, b)



x_egz = np.arange(1.0, 2.0, 0.00001)
u_egz = -1/6 * x_egz**3 + 7/6 * x_egz - 1


fn.plot(u)
plt.plot(x_egz, u_egz)

plt.xlabel('x - os')
plt.ylabel('y - os')
plt.title("Funkcija i njena FEM-aproksimacija s M = " + str(M))

plt.show()