# Python simulation of an electron in a 1d infinite box potential
# Integrate time independent SE using the Verlet method
# Locate eigenvalues by the shooting method
# MW 250402

import numpy as np
import matplotlib.pyplot as plt

h=6.62607015e-34   # Plancks constant
hbar=h/(2*np.pi)
m=9.10938356e-31   # electron mass
e=1.60217662e-19   # electron charge=-e

N=int(1e4)           # number of mesh points
a=1.0e-9           # well width a=1 nm
dx=a/N             # step length
dx2=dx**2          # step length squared
c=2.0*m/hbar**2    # constant in Schrödinger equation

# exact solution for infinite box potential
E1=h**2/(8*m*a**2)  # Joule
EeV1=E1/e            # electron volt
print('E1=',EeV1,'eV')
#print('E2=',EeV*2**2,'eV')

#Enhetslöst
N=int(1e4)           # number of mesh points
a=1.0e-9           # well width a=1 nm
dx=4/N             # step length
dx2=dx**2          # step length squared
c=2.0   # constant in Schrödinger equation

# input energy guess
#EeV = 0.3          # input energy in eV: test 0.3 , 0.4 , 0.3760 , 1.5
#E = EeV*e          # input energy in J
E = 1/2
# potential energy function
def V(x):
    #y = 0.0
    #y = x**2/2 # harmonic oscillator
    y = x**2/2 + x**4 # anharmonic oscillator
    return y

# initial values and lists
x = 0               # initial value of position x

# even solution
psi = 1.0           # wave function at initial position
dpsi = 0.0          # derivative of wave function at initial position

# odd solution
#psi = 0.0           # wave function at initial position
#dpsi = 1.0          # derivative of wave function at initial position

x_tab = []          # list to store positions for plot
psi_tab = []        # list to store wave function for plot
x_tab.append(x)
psi_tab.append(psi)

def psi_func(x, psi, dpsi, x_tab, psi_tab, N, E):
    for i in range(N) :
        d2psi = c*(V(x)-E)*psi
        psi += dpsi*dx + 0.5*d2psi*dx2
        d2psinew = c*(V(x+dx)-E)*psi
        dpsi += 0.5*(d2psi+d2psinew)*dx
        x += dx
        x_tab.append(x/a)
        psi_tab.append(psi)
    return psi

def psi_funcharmonic(x, psi, dpsi, x_tab, psi_tab, N, E):
    for i in range(N) :
        d2psi = c*(V(x)-E)*psi
        psi += dpsi*dx + 0.5*d2psi*dx2
        d2psinew = c*(V(x+dx)-E)*psi
        dpsi += 0.5*(d2psi+d2psinew)*dx
        x += dx
        x_tab.append(x)
        psi_tab.append(psi)
    return psi

psi_funcharmonic(x, psi, dpsi, x_tab, psi_tab, N, E)

#Intervallhalvering
tol = 1e-10

a = 9.3
b = 9.7
k = (a+b)/2
#while b-a > tol:
#    k = (a+b)/2
#    if psi_func(x, psi, dpsi, x_tab, psi_tab, N, a*e)*psi_func(x, psi, dpsi, x_tab, psi_tab, N, k*e) < 0:
#        b = k
#    else:
#        a = k
#E = k*e

error = [abs(-4.1123448030195087e-16), abs(-4.112334578280044e-18), abs(-4.112360603514437e-20), abs(-4.0767182960900843e-22), 1.745686970478366e-23, ]
N_vals = [1000, 1e4, 1e5, 1e6, 1e7]
energies = [0.3760301686508463, 1.5041206374906946, 3.384271295181488, 6.0164819561597795, 9.400752360653133]
energy_differences = [3.0927245209255716e-09, 4.948358878209547e-08, 2.505106491135223e-07, 7.91737353900146e-07, 1.9329361382602883e-06]

theoretical_error = []
constant = 1e-9
for i in range(0, len(N_vals)):
    theoretical_error.append(constant/(N_vals[i]**2))

plt.close()
#plt.loglog(N_vals, error, marker = 'o', linestyle='None', label = 'Uppmätt fel')
#plt.loglog(N_vals, theoretical_error, linewidth = 2, label = 'Teoretiskt fel')
#plt.xlabel("N", fontsize=15)
#plt.ylabel("Fel", fontsize = 15)
#plt.legend()
plt.plot(x_tab, psi_tab, linewidth=2)
plt.xlabel('x',fontsize=15)
plt.ylabel('$\psi$',fontsize=15)
plt.savefig('psi.pdf')
plt.show()