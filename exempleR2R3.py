# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:51:05 2022

@author: pchauris
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
#%% Def classes et fonctions
class Cell:
    def __init__(self,index,level):
        self.level = level
        self.index = index
    def info(self):
        print('index :',self.index)
        print('level :',self.level)
    def split(self):
        C00 = Cell(np.concatenate((self.index,[0,0])),self.level + 1)
        C01 = Cell(np.concatenate((self.index,[0,1])),self.level + 1)
        C10 = Cell(np.concatenate((self.index,[1,0])),self.level + 1)
        C11 = Cell(np.concatenate((self.index,[1,1])),self.level + 1)
        return C00,C01,C10,C11
    def center(self):
        index = self.index
        level = self.level
        x1 = 0
        x2 =  0
        for k in range(level):
            x1 += index[2*k]*L/2**k/N
            x2 += index[2*k+1]*L/2**k/N
        x1 += L/N/2**(k+1) + x10
        x2 += L/N/2**(k+1) + x20
        return np.array([x1,x2])

def cen(grid):
    X1 = []
    X2 = []
    for cell in grid:
        x1,x2 = cell.center()
        X1.append(x1)
        X2.append(x2)
    return np.array(X1),np.array(X2)

def init_grid(N):
    grid = []
    for i in range(N):
        for j in range(N):
            grid.append(Cell(np.array([i,j]),1))
    return grid


def crit_sequence(grid):
    res = []
    for cell in grid:
        j = critere(cell)
        res.append(j)
    return np.array(res)

def alpha_sequence(grid):
    res = crit_sequence(grid)
    return np.linspace(0,res.mean(),len(grid))

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
    #maximum global
    fmax = np.max(f)
    idmax = np.where(f == fmax)
    idmax = idmax[0][0]
    #alpha 
    alphamax = alpha[idmax]
    return alphamax

def iterate_grid(grid,alpha):
    new_grid = grid.copy()
    for cell in grid:
        k = grid.index(cell)
        J = critere(cell)
        if J > alpha:
            C00,C01,C10,C11 = cell.split()
            new_grid.remove(cell)
            new_grid.insert(k,C11)
            new_grid.insert(k,C10)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
    return new_grid

def T_direct(x1,x2):
    s = x1**2+x2**2+1
    z = np.array([s,x1/s,x2/s])
    return np.transpose(z)

def T_inverse(z1,z2,z3):
    return np.array([z1*z2,z1*z3])

def dT_dx(x1,x2):
    den = (x1**2+x2**2+1)**2
    return np.array([[2*x1,2*x2],
                     [(x2**2-x1**2+1)/den,-2*x1*x2/den],
                     [-2*x1*x2/den,(x1**2-x2**2+1)/den]])
def dT_dz(x1,x2):
    z1,z2,z3 = T_direct(x1,x2)
    return np.array([[z2,z1,0],[z3,0,z1]])

def critere(cell):
    x1,x2 = cell.center()
    J = dT_dx(x1,x2)
    #return max(np.sqrt(J[0,0]**2 + J[0,1]**2)/2.75, np.sqrt(J[1,0]**2 + J[1,1]**2), np.sqrt(J[2,0]**2 + J[2,1]**2))/2
    return np.max(np.abs(J))

#%%
N = 10
L = 2
x10,x20 = -1,-1
axe = 1

grid = init_grid(N)
X,Y = cen(grid)
J = [critere(cell) for cell in grid]

plt.figure()
plt.xlim(x10,x10+L)
plt.ylim(x20,x20+L)
plt.scatter(X,Y,c = T_direct(X,Y)[:,0],s = 2,cmap = 'jet')
plt.axis('square')
plt.colorbar()

plt.figure()
plt.xlim(x10,x10+L)
plt.ylim(x20,x20+L)
plt.scatter(X,Y,c = T_direct(X,Y)[:,1],s = 2,cmap = 'jet')
plt.axis('square')
plt.colorbar()

plt.figure()
plt.xlim(x10,x10+L)
plt.ylim(x20,x20+L)
plt.scatter(X,Y,c = T_direct(X,Y)[:,2],s = 2,cmap = 'jet')
plt.axis('square')
plt.colorbar()

plt.figure()
plt.xlim(x10,x10+L)
plt.ylim(x20,x20+L)
plt.scatter(X,Y,c = J,s = 2,cmap = 'jet')
plt.axis('square')
plt.colorbar()

#%%
for _ in range(3):
    alpha = auto_threshold(grid)
    print('alpha :',alpha)
    grid = iterate_grid(grid,alpha)
    X,Y = cen(grid)

plt.figure()
plt.xlim(x10,x10+L)
plt.ylim(x20,x20+L)
plt.scatter(X,Y,c = T_direct(X,Y)[:,0],s = 2,cmap = 'jet')
plt.axis('square')
plt.colorbar()

plt.figure()
plt.xlim(x10,x10+L)
plt.ylim(x20,x20+L)
plt.scatter(X,Y,c = T_direct(X,Y)[:,1],s = 2,cmap = 'jet')
plt.axis('square')
plt.colorbar()

plt.figure()
plt.xlim(x10,x10+L)
plt.ylim(x20,x20+L)
plt.scatter(X,Y,c = T_direct(X,Y)[:,2],s = 2,cmap = 'jet')
plt.axis('square')
plt.colorbar()

#%% apprentissage de T sur grille non uniforme
mesh = np.stack((X,Y),-1)
data = T_direct(X,Y)
# fit
mlp_reg = MLPRegressor(hidden_layer_sizes=(150,50),
                       max_iter = 300,activation = 'relu',
                       solver = 'adam')

mlp_reg.fit(mesh,data)


# prediction 
Nt = 50
x = np.linspace(-1,1,Nt)
X1,X2 = np.meshgrid(x,x)
Xt = np.stack((X2,X1),-1)
Xt = Xt.reshape(Nt*Nt,2)
y_exact = T_direct(X1,X2)
y_exact = y_exact.reshape(Nt*Nt,3)
y_pred = mlp_reg.predict(Xt)



plt.figure()
plt.scatter(X1.flatten(),X2.flatten(),c = y_exact[:,axe],cmap = 'jet',s = 1)
plt.axis('square')
plt.colorbar()

plt.figure()
plt.scatter(X1.flatten(),X2.flatten(),c = y_pred[:,axe],cmap = 'jet',s = 1)
plt.axis('square')
plt.colorbar()

err = np.abs(y_exact[:,axe] - y_pred[:,axe])
plt.figure()
plt.scatter(X1.flatten(),X2.flatten(),c = err,cmap = 'jet',s = 1)
plt.axis('square')
plt.colorbar()

print('erreur (non uniforme):', metrics.mean_squared_error(y_exact, y_pred))  

#%% apprentissage de T sur grille uniforme
N_uni = int(np.sqrt(len(grid)))
x_uni = np.linspace(x10,x10+L,N_uni)
X_uni,Y_uni = np.meshgrid(x_uni,x_uni)
X_uni = X_uni.reshape(N_uni*N_uni)
Y_uni = Y_uni.reshape(N_uni*N_uni)
grid_uni = np.stack((X_uni,Y_uni),-1)
data_uni = T_direct(X_uni,Y_uni)

# fit
mlp_reg_uni = MLPRegressor(hidden_layer_sizes=(150,50),
                       max_iter = 300,activation = 'relu',
                       solver = 'adam')

mlp_reg_uni.fit(grid_uni,data_uni)

# prediction
y_pred = mlp_reg_uni.predict(Xt)

plt.figure()
plt.scatter(X1.flatten(),X2.flatten(),c = y_exact[:,axe],cmap = 'jet',s = 1)
plt.axis('square')
plt.colorbar()

plt.figure()
plt.scatter(X1.flatten(),X2.flatten(),c = y_pred[:,axe],cmap = 'jet',s = 1)
plt.axis('square')
plt.colorbar()

err = np.abs(y_exact[:,axe] - y_pred[:,axe])
plt.figure()
plt.scatter(X1.flatten(),X2.flatten(),c = err,cmap = 'jet',s = 1)
plt.axis('square')
plt.colorbar()

print('erreur (uniforme):', metrics.mean_squared_error(y_exact, y_pred))  
#%%
x = np.linspace(0,2,10)
X,Y = np.meshgrid(x,x)
mesh = np.stack((X,Y),-1)
mesh = mesh.reshape(100,2)
plt.scatter(mesh[:,0],mesh[:,1])
plt.axis('square')