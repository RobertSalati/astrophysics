"""
Title: PHYS 333 - Astrophysics: Term Project
    
Purpose: Simulate N body problem

Created on Tue Oct 27 23:59:06 2020

Author: Robert Salati
"""

#Modules

from scipy.integrate import odeint
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#=============================================================================
#    NOTES FOR USER
#    IF YOU WOULD LIKE TO RUN SOME OF THE SIMULATIONS DISCUSSED ON YOUR OWN,
#    COPY AND PASTE THE CODE UNDER EACH TABLE IN APPENDIX B RIGHT HERE:
#-----------------------------------------------------------------------------      
n = 3                                               # Number of bodies
m = [1,1,1]                                         # Mass of bodies [mSun]
tmax1 = 2                                           # Span [yr]
r = [[0,0], [0,1], [0,-1]]                          # Position of bodies [Au]
v = [[0.001,0], [1,0], [-1,0]]                      # Velocity of bodies [vEarth]
#-----------------------------------------------------------------------------
#    TO CONTROL THE NUMBER OF STEPS PER YEAR, CHANGE THE NUMBER IN THIS LINE 
#    OF CODE BELOW: 
#-----------------------------------------------------------------------------
acc_i = 1000
#-----------------------------------------------------------------------------
#    IF YOU WANT AN ANIMATION, TYPE THE LINE animation(sol) IN THE CONSOLE  
#    THIS WILL CREATE A GIF AND SAVE IT TO YOUR DOWNLOADS FOLDER AS 
#    "Salati Animation". THIS CODE IS CAPABLE OF ANIMATING UP TO 9 BODIES.
#
#    IF YOU WANT TO CHANGE THE SMOOTHNESS OF THE ANIMATION, CHANGE THIS LINE
#    BELOW. A LOWER NUMBER IS SMOOTHER BUT TAKES LONGER TO RUN:
#-----------------------------------------------------------------------------
anistep = 100
#=============================================================================


#Code Start

# Accuracy Constants
acc = int(tmax1*acc_i)                              # number of time steps for odeint

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
tmax = tmax1 * yr                                   # Converts from [yr] to [s]


def odes(x,t):
    """Set up N number of ODEs to be passed to odeint

    Args:
        x (Array): Initial conditions. Ordered r conditions first, 
        velocity conditions second.
        t (Array): DESCRIPTION.

    Returns:
        None.

    """
    dvdt = []                                   # Empty array for acceleration vectors
    for j in range(n):                          # Creates n vectors
        dv = 0
        rj = x[2*j:2*(j+1)]                     # Sets up position vectors for the body
        for i in range(n):                      # Calculates each vector
            if i != j:
                ri = x[2*i:2*(i+1)]             # Sets up position vectors for all other bodies
                
                dv += G*m[i]*(ri-rj)/norm(ri-rj)**3 # ODE for acceleration
                
        dvdt.append(dv)
    dvdt = np.array(dvdt).flatten()             # Combine all accelerations into an array
    v = x[int(len(x)/2):len(x)]                 # Combine all velocities into an array
    f = np.concatenate([v,dvdt])                # Final array with velocities and accelerations

    return(f)

def nBodies():
    """Solve and plot ODEs.
    
    Returns:
        sol (Array): Values for position and velocity for each particle at each 
          for every time from 0 to tmax.
    """
    global r, v, m, n, tmax
    initial = np.concatenate([r[:n],v[:n]]).flatten()   # Initial conditions, position first, velocity second
    t = np.linspace(0,tmax,acc)                         # Time array
    sol,d = odeint(odes,initial,t,full_output = True)   # Solves ODEs
    for i in range(n):                                  # Plotting
        r = sol[:,2*i:2*i+2]/Au
        plt.plot(r[:,0],r[:,1],label=labels[i])
        plt.scatter(r[:,0][0],r[:,1][0])
    plt.legend()
    plt.xlabel('x position [AU]')
    plt.ylabel('y position [AU]')
    
    return sol

def animation(sol):
    """Create an animation of the bodies.
    
    Args:
        sol (Array): Values for position and velocity for each particle at each 
          for every time from 0 to tmax.

    Returns:
        None.

    """
    xmin = []
    xmax = []
    ymin = []
    ymax = []                                           # Plot bounds
    fig,ax = plt.subplots()                             # Create plot
    lines = []                                          # Array of line objects
    
    if n > 0:                                           # Body 1
        r1_array = sol[:,:2]
        r1x = r1_array[:,0]                                         # Set up x array
        r1y = r1_array[:,1]                                         # Set up y array
        x1_data=[]                                                  # Initialize arrays for points to be plotted
        y1_data=[]                                                  
        line1, = ax.plot(0,0,'o', ls='-',ms=8, markevery=[0,-1])    # Create line
        xmin.append(np.min(r1x))
        xmax.append(np.max(r1x))
        ymin.append(np.min(r1y))
        ymax.append(np.max(r1y))                                    # Min and max values for bounds
        lines.append(line1,)                                        # Add body 1 line to lines array
    
    if n > 1:                                           # Body 2   
        r2_array = sol[:,2:4]
        r2x = r2_array[:,0]
        r2y = r2_array[:,1]
        x2_data=[]
        y2_data=[]
        line2, = ax.plot(0,0,'o', ls='-',ms=8, markevery=[0,-1])
        xmin.append(np.min(r2x))
        xmax.append(np.max(r2x))
        ymin.append(np.min(r2y))
        ymax.append(np.max(r2y))
        lines.append(line2,)
    if n > 2:                                           # Body 3 
        r3_array = sol[:,4:6]
        r3x = r3_array[:,0]
        r3y = r3_array[:,1]
        x3_data=[]
        y3_data=[]
        line3, = ax.plot(0,0,'o', ls='-',ms=8, markevery=[0,-1])
        xmin.append(np.min(r3x))
        xmax.append(np.max(r3x))
        ymin.append(np.min(r3y))
        ymax.append(np.max(r3y))
        lines.append(line3,)
    if n > 3:                                           # Body 4   
        r4_array = sol[:,6:8]
        r4x = r4_array[:,0]
        r4y = r4_array[:,1]
        x4_data=[]
        y4_data=[]
        line4, = ax.plot(0,0,'o', ls='-',ms=8, markevery=[0,-1])
        xmin.append(np.min(r4x))
        xmax.append(np.max(r4x))
        ymin.append(np.min(r4y))
        ymax.append(np.max(r4y))
        lines.append(line4,)
    if n > 4:                                           # Body 5   
        r5_array = sol[:,8:10]
        r5x = r5_array[:,0]
        r5y = r5_array[:,1]
        x5_data=[]
        y5_data=[]
        line5, = ax.plot(0,0,'o', ls='-',ms=8, markevery=[0,-1])
        xmin.append(np.min(r5x))
        xmax.append(np.max(r5x))
        ymin.append(np.min(r5y))
        ymax.append(np.max(r5y))
        lines.append(line5,)
    if n > 5:                                           # Body 6   
        r6_array = sol[:,10:12]
        r6x = r6_array[:,0]
        r6y = r6_array[:,1]
        x6_data=[]
        y6_data=[]
        line6, = ax.plot(0,0,'o', ls='-',ms=8, markevery=[0,-1])
        xmin.append(np.min(r6x))
        xmax.append(np.max(r6x))
        ymin.append(np.min(r6y))
        ymax.append(np.max(r6y))
        lines.append(line6,)
    if n > 6:                                           # Body 7   
        r7_array = sol[:,12:14]
        r7x = r7_array[:,0]
        r7y = r7_array[:,1]
        x7_data=[]
        y7_data=[]
        line7, = ax.plot(0,0,'o', ls='-',ms=8, markevery=[0,-1])
        xmin.append(np.min(r7x))
        xmax.append(np.max(r7x))
        ymin.append(np.min(r7y))
        ymax.append(np.max(r7y))
        lines.append(line7,)
    if n > 7:                                           # Body 8   
        r8_array = sol[:,14:16]
        r8x = r8_array[:,0]
        r8y = r8_array[:,1]
        x8_data=[]
        y8_data=[]
        line8, = ax.plot(0,0,'o', ls='-',ms=8, markevery=[0,-1])
        xmin.append(np.min(r8x))
        xmax.append(np.max(r8x))
        ymin.append(np.min(r8y))
        ymax.append(np.max(r8y))
        lines.append(line8,)
    if n > 8:                                           # Body 9   
        r9_array = sol[:,16:18]
        r9x = r9_array[:,0]
        r9y = r9_array[:,1]
        x9_data=[]
        y9_data=[]
        line9, = ax.plot(0,0,'o', ls='-',ms=8, markevery=[0,-1])
        xmin.append(np.min(r9x))
        xmax.append(np.max(r9x))
        ymin.append(np.min(r9y))
        ymax.append(np.max(r9y))
        lines.append(line9,)
    
    xmin = np.min(xmin)
    xmax = np.max(xmax)
    ymin = np.min(ymin)
    ymax = np.max(ymax)
    ax.set_xlim(xmin-.003,xmax+.003)
    ax.set_ylim(ymin-.003,ymax+.003)                    # Set bounds
       
    def frame(i):
        """Set up FuncAnimation.
        
        Args:
            i (float): Counter for FuncAnimation.

        Returns:
            lines (Array): Array of lines to be animated.

        """
        if n > 0:                                       # Body 1
            x1_data.append(r1x[i])
            y1_data.append(r1y[i])                          # Add point to x and y arrays
            
            lines[0].set_xdata(x1_data)
            lines[0].set_ydata(y1_data)                     # Pass data to lines to be plotted
            
        if n > 1:                                       # Body 2
            x2_data.append(r2x[i])
            y2_data.append(r2y[i])    
        
            lines[1].set_xdata(x2_data)
            lines[1].set_ydata(y2_data)
            
        if n > 2:                                       # Body 3
            x3_data.append(r3x[i])
            y3_data.append(r3y[i])    
        
            lines[2].set_xdata(x3_data)
            lines[2].set_ydata(y3_data)
            
        if n > 3:                                       # Body 4
            x4_data.append(r4x[i])
            y4_data.append(r4y[i])    
        
            lines[3].set_xdata(x4_data)
            lines[3].set_ydata(y4_data)
            
        if n > 4:                                       # Body 5
            x5_data.append(r5x[i])
            y5_data.append(r5y[i])    
        
            lines[4].set_xdata(x5_data)
            lines[4].set_ydata(y5_data)
            
        if n > 5:                                       # Body 6
            x6_data.append(r6x[i])
            y6_data.append(r6y[i])    
        
            lines[5].set_xdata(x6_data)
            lines[5].set_ydata(y6_data)
            
        if n > 6:                                       # Body 7
            x7_data.append(r7x[i])
            y7_data.append(r7y[i])    
        
            lines[6].set_xdata(x7_data)
            lines[6].set_ydata(y7_data)
            
        if n > 7:                                       # Body 8
            x8_data.append(r8x[i])
            y8_data.append(r8y[i])    
        
            lines[7].set_xdata(x8_data)
            lines[7].set_ydata(y8_data)            
            
        if n > 8:                                       # Body 9
            x9_data.append(r9x[i])
            y9_data.append(r9y[i])    
        
            lines[8].set_xdata(x9_data)
            lines[8].set_ydata(y9_data)            
            
        return lines[:n]
    
    animation = FuncAnimation(fig,func=frame,frames = np.arange(0,acc,anistep),interval=100)    # Create animation
    animation.save('Salati Animation.gif',fps=15)                                               # Save animation

sol = nBodies()/Au                                  # Calls the function to calculate the ode

