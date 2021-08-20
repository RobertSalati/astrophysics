"""
Title: PHYS 333 - Astrophysics: Term Project
    
Purpose: Simulate N body problem

Created on Tue Oct 27 23:59:06 2020

Author: Robert Salati
"""

#Modules

import scipy as sci
from scipy.integrate import odeint
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


# Code Start

# System parameters

n = 3

m = [1.1, 0.907, 4, 1, 1]                                   # Mass of bodies [mSun]

tmax1 = 1                                           # Run time [years]

r = [[-0.5, 0, 0], [0.5, 0, 0], [0, 1, 0], [3, 0,0], [4, 0,0]]   # Position of bodies [Au]

v = [[0.01, 0.01, 0], [-0.05, 0, -0.1], [0, 0, 1], [0, 1, 2], [0, -1, 0]]             # Velocity of bodies [vEarth]

# Accuracy Constants
acc = int(tmax1*10000)                                         # number of time steps for odeint

# Global Constants
G = 6.67430*10**-11                                 # Gravitational constant [N*m^2/kg^2]
mSun = 1.989*10**30                                 # Mass of the sun [kg]
vEarth = 30000                                      # Orbital velocity of the earth [m/s]
Au = 1.496*10**11                                   # Length of an Au [m]
yr = 31536000                                       # Seconds in a year
labels = ['Body 1','Body 2','Body 3','Body 4','Body 5','Body 6','Body 7','Body 8','Body 9',]

# Conversions:
m = np.array(m) * mSun                              # Converts from [mSun] to [kg]
r = np.array(r) * Au                                # Converts from [Au] to [m]
v = np.array(v) * vEarth                            # Converts from [vEarth] to [m/s]
tmax = tmax1 * yr                                    # Converts from [yr] to [s]


def odes(x,t):
    """Set up N number of ODEs to be passed to odeint

    Args:
        x (Array): Initial conditions. Ordered r conditions first, 
        velocity conditions second.
        t (Array): DESCRIPTION.

    Returns:
        None.

    """
    dvdt = []
    for j in range(n):                          # Creates n vectors
        dv = 0
        rj = x[3*j:3*(j+1)]
        for i in range(n):                      # calculates each vector
            if i != j:
                ri = x[3*i:3*(i+1)]
                
                dv += G*m[i]*(ri-rj)/norm(ri-rj)**3
        dvdt.append(dv)
    dvdt = np.array(dvdt).flatten()
    v = x[int(len(x)/2):len(x)]
    f = np.concatenate([v,dvdt])

    return(f)

def nBodies():
    """Solve and plot ODEs.
    
    Returns:
        sol (Array): Values for position and velocity for each particle at each 
          for every time from 0 to tmax.
    """
    global r, v, m, n, tmax
    initial = np.concatenate([r[:n],v[:n]]).flatten()
    t = np.linspace(0,tmax,acc)
    sol = odeint(odes,initial,t)
    sol = sol/Au
    r = r/Au
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection="3d")
    for i in range(n):
        R = sol[:,3*i:3*i+3]
        ax.scatter
        ax.plot(R[:,0],R[:,1],R[:,2],label=labels[i])
        ax.scatter(r[i][0],r[i][1],r[i][2])
    plt.legend()
    
    return sol
nBodies()