# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:52:46 2022

@author: pierre chauris
"""

import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff

def f(theta,x):
    [a,b] = theta
    return a*x*np.exp(-b*x**2)

def dJ_da(theta,x):
    X = -(Y_data-f(theta,x))*x*np.exp(-b*x**2)
    return np.sum(X)/N

def dJ_db(theta,x):
    X = (Y_data-f(theta,x))*a*x**3*np.exp(-b*x**2)
    return np.sum(X)/N

def curvature(f,dx):
    dx2 = FinDiff(0,dx,2)
    dx1 = FinDiff(0,dx,1)
    curv = np.abs(dx2(f))/(1+dx1(f)**2)**(3/2)
    return curv

def decoupage(curv,level):
    test = curv>level
    res = [0]

    for i in range(1,curv.size):
        if(test[i]!=test[i-1]):
            res.append(i)
    res.append(curv.size-1)
    return res

def nu_grid(curv,level,step):
    N = curv.size
    Itot = np.sum(curv)
    index = decoupage(curv,level)
    grid = np.array([])
    Ntot = 0
    for i in range(len(index)-2):
        start = index[i]
        end = index[i+1]
        Npoints = int(np.sum(curv[start:end])/Itot*N)
        Npoints = max(1,Npoints)
        print(start,end,Npoints)
        Ntot+=Npoints
        grid = np.r_[grid,np.linspace(start*step,end*step,Npoints,endpoint=False)]
    i = i+1
    start = index[i]
    end = index[i+1]
    Npoints = N-Ntot
    #print(start,end,Npoints)
    grid = np.r_[grid,np.linspace(start*step,end*step,Npoints)]
    grid = grid/np.max(grid)*10 -5
    return grid

def nu_grid2(curv):
    densite = 1/(1+10*curv)
    Normalisation = np.sum(densite)
    densite = densite/Normalisation
    return np.cumsum(densite)*10-5

# generation des données

a = 1
b = 1
theta = [a,b]
N = 50
N_fine = 200
step = 10/N

x_fine = np.linspace(-5,5,N_fine)
f_fine = f(theta,x_fine)
dxfine = x_fine[1]-x_fine[0]
curv_fine = curvature(f_fine,dxfine)
curv_fine = curv_fine/np.max(curv_fine)
curv = curv_fine[::N_fine//N]
grid = nu_grid(curv,0.2,step)
#grid = nu_grid2(curv)

f_nu = f(theta,grid)
bruit = np.array([-0.04462941,  0.04133304, -0.12484282,  0.11365125, -0.37967774,
       -0.23689244, -0.04423668,  0.1297325 , -0.00937049,  0.14260683,
       -0.06965246,  0.11816932,  0.05572795,  0.10465766,  0.20521683,
        0.16315702, -0.2433381 ,  0.45880795, -0.05389907, -0.09326108,
       -0.28345893, -0.14161946, -0.14797975, -0.1903044 , -0.13073584,
        0.12639395,  0.15959696, -0.13316231,  0.28786889,  0.14702441,
        0.02919963,  0.11225376,  0.02157862,  0.24167026,  0.02508687,
        0.24392963, -0.02571978, -0.20950212,  0.26378531, -0.18266231,
       -0.02666276, -0.10868961,  0.1944365 ,  0.25757576,  0.17381231,
       -0.05203679,  0.14356724, -0.1242826 ,  0.02300693, -0.25252006])
Y_data = f_nu + bruit

plt.scatter(grid,Y_data,10)
plt.plot(grid,f_nu)

# descente de gradient

alpha = 0.1
theta0 = np.array([1.3,0.2])
J = np.array([dJ_da(theta0,grid),dJ_db(theta0,grid)])

Coef1 = [theta0[0]]
Coef2 = [theta0[1]]
Cost = [np.linalg.norm(J)]

niter = 0
while(np.linalg.norm(J)>0.0001):
    niter += 1
    theta0 = theta0 - alpha*J
    J = np.array([dJ_da(theta0,grid),dJ_db(theta0,grid)])
        
    Coef1.append(theta0[0])
    Coef2.append(theta0[1])
    Cost.append(np.linalg.norm(J))
    
print('nombre d iterations grille non uniforme:',niter)

# affichage résultat

im,(ax1,ax2,ax3) = plt.subplots(3,1)
ax1.scatter(grid,Y_data,10,'b')
ax1.plot(grid,f(theta0,grid),'r')
ax1.plot(grid,f(theta,grid),'b')

ax2.plot(Coef1)
ax2.plot(Coef2)

ax3.plot(Cost)