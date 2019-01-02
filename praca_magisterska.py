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

#"exp(-a*pow(x[0]-7, 2) - a*pow(x[1]-4, 2))"
#ustal początkową gęstość ludzi
rho_0 = Expression("exp(-a*pow(x[0]-7, 2) - a*pow(x[1]-4, 2))", degree=2, a=2)
#rho_0 = Constant(5)

#F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
F = u*v*dx + sigma*sigma*dot(grad(u), grad(v))*dx
a, L = lhs(F), rhs(F)

u = Function(V)
velocity = Function(V)
solve(a == L, u, bc)

#phi -> <class 'ufl.mathfunctions.Ln'>
phi = uf.ln(u)
print(type(phi))


dx0 = project(phi.dx(0))
dx1 = project(phi.dx(1))
unnormed_grad_phi = project(grad(phi))
module = sqrt(dx0*dx0+dx1*dx1)
normed_phi = unnormed_grad_phi/module


rhoold = interpolate(rho_0, V)

# rhoold -> <class 'dolfin.function.function.Function'>
print(type(rhoold))
print(type(grad(rhoold)))



plot(rhoold)
plot(normed_phi)

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 14,
        }

space = assemble(rhoold * dx(mesh))
plt.text(2, 0.65, space, fontdict=font)

plt.show()

rho = TrialFunction(V)      #unknown function
trial = TestFunction(V)     #function which multipies PDE

#-rho*(Constant(0)-rho)*grad(phi)*grad(v)*dx
#-rho*uf.diff(Constant(0), rho)*grad(phi)*grad(v)*dx

#alternatywna, poprzednia
# - velocity*dot(rho, dot(grad(phi), grad(trial)))*dx
# velocity * dot(rho, dot(grad(phi), grad(trial))) * dx \

#- div(dot(rho*velocity, grad(phi)))* trial\
# div(rho*grad(phi))
#+ div(rhoold*velocity*normed_phi)*trial*dx

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
    #velocity = 1-rho#uf.diff(uf.Constant(1.0), uf.Constant(rho))
    a, L = lhs(F), rhs(F)
    solve(a == L, rho, bc2)

    rhoold.assign(rho)

    #file = File("problem.pvd")
    #file << rho

    plot(rho, interactive=True)
    plot(normed_phi)


    #tutaj wstaw kod dotyczący wylicznia pola
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 22,
            }

    space = assemble(rho * dx(mesh))
    plt.text(2, 0.65, space, fontdict=font)

    plt.show()

#wyświetlenie już samej dywergencji z gęstości ludzi
plot(div(rhoold*velocity*normed_phi))
plt.show()