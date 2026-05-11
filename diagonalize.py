
"""
Matrix diagonalization sample code
MW version 250402
"""

import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt

dim = 10 # truncated dimension of Hilbert space

def delta(m,n):
    d = 0
    if m==n: d=1
    return d

def xm(m,n):
    return (np.sqrt(n+1)*delta(m,n+1)+np.sqrt(n)*delta(m,n-1))/np.sqrt(2)

def pm(m,n):
    return 1j*(np.sqrt(n+1)*delta(m,n+1)-np.sqrt(n)*delta(m,n-1))/np.sqrt(2)

x = np.zeros(dim*dim, dtype=complex).reshape(dim, dim)
p = np.zeros(dim*dim, dtype=complex).reshape(dim, dim)
for i in range(dim):
    for j in range(dim):
        x[i,j] = xm(i,j)
        p[i,j] = pm(i,j)
x2 = x.dot(x)
p2 = p.dot(p)
#x4 = x2.dot(x2) # quartic term

# Harmonic Hamiltonian
H = p2/2+x2/2
E = LA.eigvalsh(H)
#print('E=',E[:10])
n = np.arange(dim)
Eexact = n+1/2
#Hdiag = np.real(np.diag(H))
#print('E=',Hdiag[:10])

# plot eigenvalues vs n up to nmax
nmax = 10
if nmax>dim: nmax = dim
n = np.arange(dim)
plt.plot(n[:nmax], E[:nmax], 'r.-')
plt.plot(n[:nmax], Eexact[:nmax], 'b.')
plt.xlabel('$n$',fontsize=15)
plt.ylabel('$E_n$',fontsize=15)
#plt.savefig('eigenvalues.pdf')
plt.show()


