#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 17:10:50 2020

@author: luismiguells
"""

"""
Test functions for optimization
Ackley function
-20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.exp(1) + 20

Beale function
(1.5-x+(x*y))**2 + (2.25*-x+(x*y**2))**2 + (2.625-x+(x*y**3))**2

Booth function
(x+(2*y)-7)**2 + ((2*x)+y-5)**2   

Himmelblau's function
(x**2+y-11)**2 + (x+y**2-7)**2

McCormick function
np.sin(x+y) + (x+y)**2 - 1.5*x + 2.5*y + 1

More in: https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import random
import numpy as np
import matplotlib.pyplot as plt


def obj_fun(X):
    """
    Parameters
    ----------
    X : An array that contains values x1 and x2

    Returns
    -------
    Evaluate a test function. For example, Booth function.
    """
    x = X[0]
    y = X[1]
    
    return 0.26*(x**2+y**2)-0.48*x*y # Change for the desire function

# Initialization
a = -10
b = 10
n = 50 # Number of particles
max_iter = 500
ite = 1
neighbors = 5

# Initialize particles position randomly
X = np.array([np.array([random.uniform(a, b), random.uniform(a, b)]) 
                        for _ in range(n)])
Y = 0 # Target value
epsilon = 1e-6

# PSO constants
C1 = 0.5
C2 = 0.5
W = 0.5
V = 0.05 


while ite < max_iter:
    # Evaluate the particle position in the objective function
    loc = np.array([obj_fun(X[i]) for i in range(n)])
    
    # Obtain the best global postion 
    q = np.sort(loc)
    indx0 = np.argsort(loc)
    g_best = X[indx0[0]]

    error = abs(obj_fun(g_best) - Y)

    if(error < epsilon):
        break
    
    for k in range(n):
        
        # Calculate the distance and obtain the best distance
        dist = np.sqrt(np.sum(X[k]-X, axis=1)**2)
        q = np.sort(dist)
        indx = np.argsort(dist)
        
        X_N = X[indx[0:neighbors]] # Best neighbors
        L_N = loc[indx[0:neighbors]] # Indexes of the best neighbors
        
        # Get the best particle position
        q = np.sort(L_N)
        indx1 = np.argsort(L_N)
        p_best = X_N[indx1[0]]
        
        # Calcualte the velocity of the particle and update its position
        V = W*V + ((C1*random.random())*(p_best-X[k])) \
            + ((C2*random.random())*(g_best-X[k]))
        X[k] = X[k] + V
    
    ite += 1 

print("The minimum value is in (X, Y)", g_best[0], g_best[1], "in", ite, "iterations")

# Set the values for the plot the figure
ax = plt.axes(projection='3d')
x = np.linspace(a, b, 100)
y = np.linspace(a, b, 100)
X, Y = np.meshgrid(x, y)

points = np.stack([X, Y])
Z = obj_fun(points)

# Plot the figure
ax.plot_wireframe(X, Y, Z, linewidth=0.2)

# Plot the minimum value
ax.scatter(g_best[0], g_best[1], obj_fun(g_best), 'o', color='red')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

