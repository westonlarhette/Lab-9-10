#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:34:18 2024

@author: westonlarhette
"""
""" LAB 9 & 10 """

"""More on histograms"""
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random as rng

data = rng(1000)

counts, bin_edges, _ = plt.hist(data)
#counts, bin_edges = np.histogram(data)

bin_size = bin_edges[1] - bin_edges[0]
new_widths = bin_size * counts / counts.max()

plt.figure()
plt.bar(bin_edges[:-1], counts, width=new_widths, color=['r','g','b','y'])

# =============================================================================

""" Contour plots, Surface plots and heat maps"""
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

# Create grid of x and y coordinates

x_vals = np.linspace(-1,1,200)
y_vals = np.linspace(-1,1,200)
X, Y = np.meshgrid(x_vals,y_vals)

# Generate function values

Z = np.cos(X) * np.sin(Y)

R = X**2 + Y**2

# Plot contours
plt.contour(X,Y,R)
plt.show()

plt.contour(X,Y,Z)
plt.show()

#Plot and label contours.
plt.figure()
cs = plt.contour(X,Y,Z,10,linewidths=3,colors='r')
plt.clabel(cs,fontsize=10)
plt.show()


from mpl_toolkits.mplot3d import Axes3D # Import 3D plotting tool
# Plot surfaces
ax1 = plt.axes(projection = '3d')
ax1.plot_surface(X,Y,R)
plt.show()

ax3 = plt.axes(projection='3d')
ax3.plot_surface(X,Y,R, rcount = 200, ccount = 200)
plt.show()

plt.pcolormesh(X,Y,R)
plt.show()

plt.pcolormesh(X,Y,R,cmap='jet')
# =============================================================================


"""" Lab 9 """
""" Example 1: 3D Random Walk"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed = 42)

num_steps = 500
x_step = rng.random(num_steps) > 0.5
y_step = rng.random(num_steps) > 0.5
z_step = rng.random(num_steps) > 0.5

x_step = 2*x_step - 1
y_step = 2*y_step - 1
z_step = 2*z_step - 1

x_position = np.cumsum(x_step)
y_position = np.cumsum(y_step)
z_position = np.cumsum(z_step)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

ax1.plot(x_position, y_position, z_position)

# Labeling plots
ax1.set_xlabel('X position') 
ax1.set_ylabel('Y position')
ax1.set_zlabel('Z position')
ax1.set_title('3D Random Walk')

plt.show()


""" Example 2: Numerical integration
a. Integrate f(x) = x^2 from 0 to 2 and check results
b. Integrate e^(-x^2 /2) from 0 to 5 and plot results
c. Can quad handle infinite limits? Use -np.inf and np.inf as limits to evaluate
the integral of the previous function from -inf to +inf. Compare to the exact result:
    sqrt(2pi)
    """
#import numpy.py as np
import matplotlib as plt
from scipy.integrate import quad

# Part a:
def f(x):
    return x**2
results,error = quad(f,0,2)
print("Answer:",results)
print("Error:",error)

# Part b:

def f(x):
    return np.exp(-x**2 / 2)

upper_limit = np.linspace(0,5,50)
integral = np.zeros(upper_limit.size)
for i in range(upper_limit.size):
    integral[i],error = quad(f,0,upper_limit[i])
## I am not sure how to plot these results

# Part c:
answer,error = quad(f,-np.inf,np.inf)
print("Answer:",answer)
print("Error:",error)
# The answer given is correct















