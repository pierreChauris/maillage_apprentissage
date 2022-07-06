# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:11:24 2022

@author: pchauris
"""

import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.signal import argrelextrema



def f(X):
    return 2*np.exp(-7*(X-8)**2) + np.exp(-0.8*(X-2)**2)
    return np.exp(-0.5*(X-5)**2)

def emp_grad(f,xmin,xmax):
    Df = f(xmax)-f(xmin)
    Dx = xmax-xmin
    return np.abs(Df/Dx)

def init_mesh(a,b,p0):
    dx = (b-a)/(2**p0)
    grid = np.arange(a,b,dx)
    return np.append(grid,b)

def crit_sequence(grid):
    res = []
    for k in range(grid.size-1):
        crit = emp_grad(f,grid[k],grid[k+1])
        res.append(crit)
    return np.array(res)

def alpha_sequence(grid):
    res = crit_sequence(grid)
    #return np.linspace(0,max(res),grid.size)
    return np.linspace(0,res.mean(),grid.size)

def distrib_sequence(grid):
    alpha = alpha_sequence(grid)
    crit = crit_sequence(grid)
    res = []
    for j in range(alpha.size):
        dj = np.count_nonzero(crit>alpha[j])
        res.append(dj)
    return np.array(res)
    
def auto_threshold(grid):
    alpha = alpha_sequence(grid)
    d = distrib_sequence(grid)
    f = alpha*d
    #maximum local
    indices = argrelextrema(f, np.greater)
    idmax_loc = indices[0][0]
    #macimum global
    fmax = np.max(f)
    idmax = np.where(f == fmax)
    idmax = idmax[0][0]
    #alpha 
    alphamax = alpha[idmax]
    return alphamax

def iterate_mesh(mesh):
    new_mesh = [mesh[0]]
    alpha = auto_threshold(mesh)
    print('alpha :',alpha)
    k = 0
    while k < mesh.size-2:
        Sk = emp_grad(f,mesh[k],mesh[k+1])
        if Sk > alpha:
            'raffinement'
            new_point = (mesh[k+1] - mesh[k])/2 + mesh[k]
            new_mesh.append(new_point)
            k += 1
            new_mesh.append(mesh[k])   
        if Sk < alpha :
            'fusion'
            k += 2
            new_mesh.append(mesh[k])

    if mesh[-1] not in new_mesh:
        new_mesh.append(mesh[-1])
    return np.array(new_mesh)

# uniform base grid

N = 200
a,b = 0,10
x = np.linspace(a,b,N)
y = f(x)

d_dx = FinDiff(0,x[1]-x[0],1)
df_dx = d_dx(y)

# adaptative grid

p0 = 6
niter = 4

grid = init_mesh(a,b,p0)

alpha = alpha_sequence(grid)
d = distrib_sequence(grid)
img,(ax1,ax2) = plt.subplots(2,1)
ax2.plot(alpha,d*alpha)
ax1.plot(alpha,d)
plt.show()


Alpha = []
for i in range(niter):
    plt.figure()
    plt.plot(grid,np.zeros(grid.size),'-o')
    Alpha.append(auto_threshold(grid))
    plt.plot(x,y)
    plt.plot(grid,f(grid),'o')
    print("taille de la grille :",grid.size)
    grid = iterate_mesh(grid)


fnu = f(grid)
d_dx = FinDiff(0,grid,1)
df_dx_nu = d_dx(fnu)
plt.figure()
plt.plot(grid,df_dx_nu,'--o')
plt.plot(x,df_dx)