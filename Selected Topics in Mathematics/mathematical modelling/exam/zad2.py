import fenics as fn
import matplotlib.pyplot as plt

M = int(input("M = "))

p0 = fn.Point(1, 2)
p1 = fn.Point(2, 3)
mesh = fn.RectangleMesh(p0, p1, M, M)
V = fn.FunctionSpace(mesh, 'CG', 1)

u_D = fn.Expression('0', degree=2)

def boundary(x, on_boundary):
	return on_boundary

bc = fn.DirichletBC(V, u_D, boundary)

# Slaba formulacija
u = fn.TrialFunction(V)
v = fn.TestFunction(V)
f = fn.Expression("x[0] * x[1]", degree=2)
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

# Grafovi
fig1 = plt.figure()
fn.plot(mesh)
plt.savefig('mesh2D.png')

fig2 = plt.figure()
fn.plot(u)
plt.savefig('poisson2D.png')

# File za paraview
#vtkfile = fn.File('poisson2D.pvd')
#vtkfile << u
