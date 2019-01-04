from fenics import *
import matplotlib.pyplot as plt
import mshr as ms
import ufl as uf
import numpy as np
from dolfin import *


T = 1000
num_steps = 5000
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


drzwi1 = 3.0
drzwi2 = 4.0
drzwi3 = 3.0
korytarz = 80.0

#mesh_lotnisko - hala przylotów
hala = ms.Rectangle(Point(0.0, 0.0), Point(100.0, 40.0))
stoisko1 = ms.Rectangle(Point(15.0, 18.0), Point(15.0+4.0, 18.0+5.0))
stoisko2 = ms.Rectangle(Point(80.0, 18.0), Point(80.0+4.0, 18.0+5.0))
ławka = ms.Rectangle(Point(48.0, 20.0), Point(48.0+4.0, 20.0+1.0))
mesh_lotnisko = ms.generate_mesh(hala - stoisko1 - stoisko2 - ławka, 100)

korytarz1 = ms.Rectangle(Point(15.0, 40.0), Point(15.0 + drzwi1, 40.0+korytarz))
korytarz2 = ms.Rectangle(Point(48.0, 40.0), Point(48.0 + drzwi2, 40.0+korytarz))
korytarz3 = ms.Rectangle(Point(82.0, 40.0), Point(82.0 + drzwi3, 40.0+korytarz))
#mesh_lotnisko2 - hala odlotów
mesh_lotnisko2 = ms.generate_mesh(hala - stoisko1 - stoisko2 - ławka + korytarz1 + korytarz2 + korytarz3, 100)


V = FunctionSpace(mesh_lotnisko2, "P", 1)

def ściany(x, on_boundary):
    return on_boundary and not near(x[0], długość) and \
           between(x[1], szerokość/2+drzwi/2, szerokość/2-drzwi/2)

def wyjście(x, on_boundary):
    return on_boundary and (near(x[0], długość) and x[1] < szerokość/2+drzwi/2 and x[1] > szerokość/2-drzwi/2)

def ściany_lotnisko(x, on_boundary):
    return on_boundary and not (near(x[1], 40.0+korytarz))

def wyjście_lotnisko(x, on_boundary):
    return on_boundary and (near(x[1], 40.0+korytarz))

bc = DirichletBC(V, Constant(1), wyjście)
bc2 = DirichletBC(V, Constant(0), ściany)
bc_lotnisko = DirichletBC(V, Constant(1), wyjście_lotnisko)
bc2_lotnisko = DirichletBC(V, Constant(0), ściany_lotnisko)

u = TrialFunction(V)
v = TestFunction(V)
phi = TestFunction(V)

rho_0_lotnisko_inne = Expression("exp(-a*pow(x[0]-7, 2) - a*pow(x[1]-4, 2))", degree=2, a=0.01)
rho_0_lotnisko = Expression("x[1] < 35 ? 1.0 : 0.0", degree=2, a=0.01)
rho_0 = Expression("exp(-a*pow(x[0]-7, 2) - a*pow(x[1]-4, 2))", degree=2, a=1)

F = u*v*dx + sigma*sigma*dot(grad(u), grad(v))*dx
a, L = lhs(F), rhs(F)

u = Function(V)
velocity = Function(V)
solve(a == L, u, bc_lotnisko)

phi = uf.ln(u)


dx0 = project(phi.dx(0))
dx1 = project(phi.dx(1))
unnormed_grad_phi = project(grad(phi))
module = sqrt(dx0*dx0+dx1*dx1)
normed_phi = unnormed_grad_phi/module

plot(normed_phi)
plt.show()

rhoold = interpolate(rho_0_lotnisko, V)

plot(rhoold)
plt.show()

rho = TrialFunction(V)      #unknown function
trial = TestFunction(V)     #function which multipies PDE

F = rho*trial*100.0*dx \
    - rhoold*trial*100.0*dx \
    - rhoold*exp(-rhoold*rhoold/100)*dot(normed_phi, grad(trial))*dx
    #+ sigma*dot(grad(rho), grad(trial))*dx


rho = Function(V)
t = 0
for n in range(num_steps):

    if(n%500==0):
        p = plot(rho, interactive=True)
        plot(normed_phi, linewidth=4)

        #tutaj wstaw kod dotyczący wylicznia pola
        font = {'family': 'serif',
                'color': 'white',
                'weight': 'normal',
                'size': 16,
                }

        space = assemble(project(rho, V) * dx(mesh_lotnisko))
        plt.text(2, 0.35, space, fontdict=font)

        plt.colorbar(p)
        plt.show()

    # Update current time
    t += dt

    # Compute solution
    a, L = lhs(F), rhs(F)
    solve(a == L, rho, bc2_lotnisko)

    rhoold.assign(rho)

    #file = File("problem.pvd")
    #file << rho
