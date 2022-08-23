# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:00:31 2022

@author: pchauris
"""

import numpy as np
import matplotlib.pyplot as plt
from raffinement import *
import time

def f(X,Y):
    # return Y*np.sin(X)
    return np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2)



geometry = [10,10,100,100,0,0]
# geometry = [2*np.pi,1,20,20,0,0]
grid = init_grid(geometry)
X,Y = coordinate(grid)
plt.scatter(X,Y,c=f(X,Y),cmap='jet')
plt.axis('square')
plt.colorbar()

start = time.time()
grad = []
nx,ny = 20,20
Coeffs = coeffs(grid,f(X,Y),nx,ny)
for cell in grid:
    gx,gy = gradient(cell,nx,ny,Coeffs)
    grad.append(np.sqrt(gx**2+gy**2))
    
end = time.time()
print('temps de calcul du gradient par surrogate :',end-start)
plt.figure()
plt.scatter(X,Y,c=grad,cmap='jet')
plt.axis('square')
plt.colorbar()

#%%
Z = f(X,Y)
nx,ny = 4,2
plt.scatter(X,Y,c='white')
X0,Y0,Z0 = split_grid(grid,Z,nx,ny,1,0)
plt.scatter(X0,Y0,c=Z0,cmap=('jet'))
plt.axis('square')
plt.colorbar()
#%% iteration de raffinement

grid = iterate_grid(grid, f(X,Y), False)
X,Y = coordinate(grid)
plt.scatter(X,Y,c=f(X,Y),cmap='jet',s=1)
for cell in grid:
    plot_cell(cell)
plt.axis('square')
plt.colorbar()

#%% gradient
grad = []
nx,ny = 2,2
Coeffs = coeffs(grid,f(X,Y),nx,ny)
for cell in grid:
    gx,gy = gradient(cell,nx,ny,Coeffs)
    grad.append(np.sqrt(gx**2+gy**2))

plt.figure()
plt.scatter(X,Y,c=grad,cmap='jet',s=1)

for cell in grid:
    plot_cell(cell)
plt.axis('square')
plt.colorbar()