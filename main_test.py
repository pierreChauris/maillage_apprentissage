# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:52:49 2022

@author: pchauris
"""

from raffinement import *
import matplotlib.pyplot as plt

def f_test(X,Y):
    # return np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2)
    return Y*np.sin(X)

#%% test du raffinement
geometry = [2*np.pi,1,20,11,0,0]
# geometry = [10,10,10,10,0,0]
grid = init_grid(geometry)

niter = 3

for _ in range(niter):
    X1,X2 = coordinate(grid)
    Z = f_test(X1,X2)
    print('taille de la grille :',len(grid))
    grid = iterate_grid(grid,Z,True)
    
print('taille de la grille :',len(grid))
X1,X2 = coordinate(grid)

plt.figure()
plt.scatter(X1,X2,c = f_test(X1,X2),s = 1,cmap = 'jet')
plt.colorbar()
if geometry[0]==geometry[1]:
    plt.axis('square')
    
#%% comparaison du gradient

geometry = [10,10,50,50,0,0]
grid = init_grid(geometry)
X,Y = coordinate(grid)
Z = f_test(X,Y)

grad = []
exact_grad = []
nx,ny = 10,10
Coeffs = coeffs(grid,Z,nx,ny)
for cell in grid:
    gx,gy = gradient(cell,nx,ny,Coeffs)
    grad.append(np.sqrt(gx**2+gy**2))
    gx,gy = emp_grad(cell,f_test)
    exact_grad.append(np.sqrt(gx**2+gy**2))

erreur = np.linalg.norm(np.array(grad) - np.array(exact_grad))
print('erreur :',erreur)

plt.figure()
plt.scatter(X,Y,c = grad,cmap = 'jet')
plt.colorbar()
plt.title('gradient surrogate model de degré 4')
if geometry[0]==geometry[1]:
    plt.axis('square')

plt.figure()
plt.scatter(X,Y,c = exact_grad,cmap = 'jet')
plt.colorbar()
plt.title('gradient exact')
if geometry[0]==geometry[1]:
    plt.axis('square')

#%% test R2 -> R3

def f2d(X,Y):
    return np.stack((np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2),Y*np.sin(X)),-1)

geometry1 = [10,10,10,10,0,0]
geometry2 = [2*np.pi,1,10,10,0,0]
grid1 = init_grid(geometry1)
grid2 = init_grid(geometry2)
X,Y = coordinate(grid1)
Z = f2d(X,Y)

for i in range(2):
    plt.figure()
    plt.scatter(X,Y,c = Z[:,i],s=1)
    
sup_grid = [grid1,grid2]
# iteration de raffinement
for i in range(len(sup_grid)):
    sup_grid[i] = iterate_grid(sup_grid[i],Z[:,i],True)

for gr in sup_grid:
    x,y = coordinate(gr)
    plt.figure()
    plt.scatter(x,y,s=1)
# calcul des nouveaux Z
x,y = coordinate(sup_grid[0])
Z1 = f2d(x,y)[:,0]
x,y = coordinate(sup_grid[1])
Z2 = f2d(x,y)[:,1]
# iteration de raffinement
sup_grid[0] = iterate_grid(sup_grid[0],Z1,True)
sup_grid[1] = iterate_grid(sup_grid[1],Z2,True)

# calcul des nouveaux Z
x,y = coordinate(sup_grid[0])
Z1 = f2d(x,y)[:,0]
plt.figure()
plt.scatter(x,y,c=Z1,s=1)

x,y = coordinate(sup_grid[1])
Z2 = f2d(x,y)[:,1]
plt.figure()
plt.scatter(x,y,c=Z2,s=1)

# iteration de raffinement
sup_grid[0] = iterate_grid(sup_grid[0],Z1,False)
sup_grid[1] = iterate_grid(sup_grid[1],Z2,False)


for gr in sup_grid:
    x,y = coordinate(gr)
    plt.figure()
    plt.scatter(x,y,s=1)
