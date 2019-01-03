from fenics import *
import matplotlib.pyplot as plt
import mshr as ms
import ufl as uf
import numpy as np
from dolfin import *


T = 1000
num_steps = 500
dt = T / num_steps
alpha = 0
beta = 1.2
sigma = 1000.0

# Create mesh and define function space
długość = 10.0
szerokość = 5.0
drzwi = 1.5

pomieszczenie = ms.Rectangle(Point(0.0, 0.0), Point(długość, szerokość))
przeszkoda_kolista = ms.Circle(Point(długość - szerokość/2, szerokość/2), szerokość/4)
mesh = ms.generate_mesh(pomieszczenie - przeszkoda_kolista, 100)

mesh2 = ms.generate_mesh(ms.Rectangle(Point(0.0, 0.0), Point(2.0, 2.0)), 200)

V = FunctionSpace(mesh, "P", 1)

def ściany(x, on_boundary):
    return on_boundary and not (near(x[0], długość) and x[1] < szerokość/2+drzwi/2 and x[1] > szerokość/2-drzwi/2)

def wyjście(x, on_boundary):
    return on_boundary and (near(x[0], długość) and x[1] < szerokość/2+drzwi/2 and x[1] > szerokość/2-drzwi/2)

bc = DirichletBC(V, Constant(1), wyjście)
bc2 = DirichletBC(V, Constant(0), ściany)

u = TrialFunction(V)
v = TestFunction(V)
phi = TestFunction(V)

rho_0 = Expression("exp(-a*pow(x[0]-7, 2) - a*pow(x[1]-4, 2))", degree=2, a=2)

F = u*v*dx + sigma*sigma*dot(grad(u), grad(v))*dx
a, L = lhs(F), rhs(F)

u = Function(V)
velocity = Function(V)
solve(a == L, u, bc)

phi = uf.ln(u)


dx0 = project(phi.dx(0))
dx1 = project(phi.dx(1))
unnormed_grad_phi = project(grad(phi))
module = sqrt(dx0*dx0+dx1*dx1)
normed_phi = unnormed_grad_phi/module


rhoold = interpolate(rho_0, V)


rho = TrialFunction(V)      #unknown function
trial = TestFunction(V)     #function which multipies PDE

F = rho*trial*dx \
    - rhoold*trial*dx \
    + rhoold*dot(normed_phi, grad(trial))*dx
    #+ sigma*dot(grad(rho), grad(trial))*dx

rho = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    velocity = 1 - rhoold
    a, L = lhs(F), rhs(F)
    solve(a == L, rho, bc2)

    rhoold.assign(rho)

    #file = File("problem.pvd")
    #file << rho

    p = plot(rho, interactive=True)
    plot(normed_phi)

    #tutaj wstaw kod dotyczący wylicznia pola
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 22,
            }

    space = assemble(project(rho, V) * dx(mesh2))
    plt.text(2, 0.65, space, fontdict=font)

    plt.colorbar(p)
    plt.show()
