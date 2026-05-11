# -*- coding: utf-8 -*-
"""
Split Operator Method for simulating the time dependent Schrodinger equation
with real time animation

The code solves the time dependent SE, animates the solution, 
and stores the final frame in the file plot.png
The initial state is here a Gaussian wave packet.
Any potential can be investigated.

MW version 250402 
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft,ifft,fftshift      # fast Fourier transforms
 
# set x-axis scale
N = 2**13    # choice suitable for fft
dx = 0.1
L = N*dx
x = dx*(np.arange(N)-0.5*N)

# set momentum scale
dk = 2*np.pi/L
k = -N*dk/2 + dk*np.arange(N)

# time parameters  
t = 0.0                         # start time
dt = 0.01                       # time step
tmax = 100.                    # max time
nsteps = 50                     # number of time steps between frame updates
frames = int(tmax/(nsteps*dt))

# parameters for potential barrier
w = 2.0                         # width of potential
V0 = 1.0                        # height of potential

# quantum parameters
hbar = 1.0
p = hbar*k 
m = 1.0

# parameters for the initial gaussian wave packet
a = 8.0                         # width of gaussian
x0 = -100.0                     # initial center position
E = 1.0                         # energy
k0 = np.sqrt(2*m*E)/hbar        # wavevector of wavepacket motion
print('k0,E=',k0,E) 

######################################################################
# Gaussian wave packet of width a, centered at x0, with momentum k0 
def gaussian(x, a, x0, k0):
    return ((a*np.sqrt(np.pi))**(-0.5)
            * np.exp(-0.5*((x-x0)*1./a)**2 + 1j*x*k0))
######################################################################
# Free Gaussian wave packet of width a, centered at x0, with momentum k0
# at time t
def exactgaussian(x, a, x0, k0, t):
    gamma = np.sqrt(1+1j*t/(m*a**2))
    return ((1./(a**2*np.pi))**0.25/gamma*np.exp(-(x-x0-k0*t/m)**2/
    (2.*a**2*gamma**2)+1j*k0*(x-x0-k0*t/2./m)))
######################################################################
def theta(x):    # Heaviside function
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y
######################################################################
def potential(x):   # potential that can be selected to be any function
    pot = 0*x        # free particle
    #pot[x > 0] = 100. # potential wall 
    #pot = V0*theta(x)  # potential step
    #pot = V0*(theta(x+w/2.0)-theta(x-w/2.0))  # square barrier
    #pot = V0*theta(x-1)/(x+1.e-10)  # Gamow Coulomb barrier
    return pot      
######################################################################

# Define arrays of potential energy and time evolution operators
pot = potential(x)     # potential energy
expV_half = np.exp(-1j*pot*dt/2/hbar)           # time evolution from potential energy
expV = expV_half*expV_half                   # time evolution from potential energy
expT = fftshift(np.exp(-1j*p*p*dt/(2*m)/hbar))  # time evolution from kinetic energy
# (fftshift stores the momenta in normal order, see the scipy documentation)

# calculate initial wave function
psi = gaussian(x,a,x0,k0)                    # initial wave packet
psix=exactgaussian(x,a,x0,k0,t)
#psix=exactgaussian(x,a,x0,k0,t)-exactgaussian(x,a,-x0,-k0,t) # mirror image

# set up the figure and the plot element to animate
fig = plt.figure(figsize=(12,8),dpi=80)
plt.xlim(-100,100)
plt.ylim(0,0.1)
plt.xlabel('x', fontsize=15)
plt.ylabel('$|\Psi(x,t)|^2$', fontsize=15)
psi2_curve, = plt.plot([], [], c='b', lw=4)
psix2_curve, = plt.plot([], [], c='r', linestyle='dashed', lw=4)
# plot potential
potential_curve, = plt.plot(x, 0.05*pot, c='black', lw=6) 

# calculations
def step():
    global psi,psix,t
    for it in range(nsteps) :              # don't plot all steps to speed up animation
        psi = ifft(expT*fft(expV*psi))     # one time step
    t += nsteps*dt
    psix=exactgaussian(x,a,x0,k0,t)
    #psix=exactgaussian(x,a,x0,k0,t)-exactgaussian(x,a,-x0,-k0,t) # mirror image
    #print(t)
                                                                                                                
# initialization function: plot the background of each frame
def init():
    psi2_curve.set_data([], [])
    psix2_curve.set_data([], [])
    return psi2_curve, psix2_curve 

# animation function called sequentially
def animate(i):
    step() 
    psi2_curve.set_data(x, abs(psi)**2)
    psix2_curve.set_data(x, abs(psix)**2)
    return psi2_curve, psix2_curve 

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=10, repeat=False)
plt.show()

# Save final frame
#fig = plt.figure(figsize=(12,8),dpi=80)
#plt.xlim(-100,100)
#plt.ylim(0,0.1)
#plt.plot(x, abs(psi)**2, c='b', lw=4)
#plt.plot(x, abs(psix)**2, c='r', linestyle='dashed', lw=4)
#plt.plot(x, 0.05*pot, c='black', lw=6) 
#plt.savefig('./fig.pdf', bbox_inches='tight')
#plt.show()

# calculate T and R for final state
T=np.sum(abs(psi[N//2:N])**2)
R=np.sum(abs(psi[0:N//2])**2)
# normalize to T+R=1
A=T+R
T/=A
R/=A
print('T,R=',T,R)



