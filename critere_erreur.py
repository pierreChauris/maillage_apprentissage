# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:59:00 2022

@author: pchauris
"""

"Test d'un nouveau critère basé sur l'erreur de prédiction"

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


#%% fonctions
class Cell:
    def __init__(self,x,y,geometry):
        
        Lx, Ly, Nx, Ny, Ox, Oy = geometry
        px,py = Lx/Nx,Ly/Ny

        self.geometry = geometry
        self.center = np.array([x,y])
        self.size = np.array([px,py])
        
    def info(self):
        print('index :',self.index)
        print('center :',self.center)
        print('size :',self.size)
        
    def split_iso(self):
        px,py = self.size
        #create 4 new cells
        x,y = self.center - np.array([px/4,py/4])
        C00 = Cell(x,y,self.geometry)
        C00.size = self.size/2
        
        x,y = self.center + np.array([px/4,-py/4])
        C01 = Cell(x,y,self.geometry)
        C01.size = self.size/2
        
        x,y = self.center + np.array([-px/4,py/4])
        C10 = Cell(x,y,self.geometry)
        C10.size = self.size/2
        
        x,y = self.center + np.array([px/4,py/4])
        C11 = Cell(x,y,self.geometry)
        C11.size = self.size/2
        
        return C00,C01,C10,C11


def init_grid(geometry):
    grid = []
    Lx, Ly, Nx, Ny, Ox, Oy = geometry
    x,y = np.linspace(Ox,Ox+Lx,Nx), np.linspace(Oy,Oy+Ly,Ny)
    X,Y = np.meshgrid(x,y)
    X,Y = X.flatten(),Y.flatten()
    for i in range(X.size):
        x,y = X[i],Y[i]
        grid.append(Cell(x,y,geometry))
    return grid


def coordinate(grid):
    X1 = []
    X2 = []
    for cell in grid:
        x1,x2 = cell.center
        X1.append(x1)
        X2.append(x2)
    return np.array([np.array(X1),np.array(X2)])

def plot_cell(cell):
    x,y = cell.center
    px,py = cell.size
    plt.vlines(x = x-px/2, ymin = y-py/2, ymax = y+py/2,linewidth=1)
    plt.vlines(x = x+px/2, ymin = y-py/2, ymax = y+py/2,linewidth=1)
    plt.hlines(y = y-py/2, xmin = x-px/2, xmax = x+px/2,linewidth=1)
    plt.hlines(y = y+py/2, xmin = x-px/2, xmax = x+px/2,linewidth=1)
    

def crit_sequence(grid):
    _,_,_,erreur = apprentissage(grid)
    return erreur

def alpha_sequence(grid):
    sequence = crit_sequence(grid)
    return np.linspace(0,sequence.max(),sequence.size)


def distrib_sequence(grid):
    alpha = alpha_sequence(grid)
    crit = crit_sequence(grid)
    distribution = []
    for j in range(alpha.size):
        dj = np.count_nonzero(crit>alpha[j])
        distribution.append(dj)
    return np.array(distribution)


def auto_threshold(grid):
    alpha = alpha_sequence(grid)
    distribution = distrib_sequence(grid)
    f = alpha*distribution
    #maximum global
    fmax = np.max(f)
    idmax = np.where(f == fmax)
    idmax = idmax[0][0]
    #alpha 
    alphamax = alpha[idmax]
    return alphamax

def iterate_grid(grid):
    alpha = auto_threshold(grid)
    X,Y,_,crit = apprentissage(grid)
    grid = data_to_grid(X,Y)
    new_grid = grid.copy()
    for cell in grid:
        k = grid.index(cell)        
        # raffinement iso
        if crit[k] > alpha :
            C00,C01,C10,C11 = cell.split_iso()
            if cell in new_grid:
                new_grid.remove(cell)
            new_grid.insert(k,C11)
            new_grid.insert(k,C10)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
        
    return new_grid

def apprentissage(grid):
    # données d'entrainement
    X,Y = coordinate(grid)
    train_X = np.stack((X,Y),-1)
    train_Z = f(X,Y)
    
    data_in = train_X
    data_out = train_Z
    # fit
    mlp_reg = MLPRegressor(hidden_layer_sizes=(40,20),
                           max_iter = 1000,activation = 'relu',
                           solver = 'adam')

    mlp_reg.fit(data_in,data_out)
    
    # predict
    Xtest,Ytest = uniform_grid(grid[0].geometry,2500)
    Test = np.stack((Xtest,Ytest),-1)
    Z_test = f(Xtest,Ytest)
    Z_predict = mlp_reg.predict(Test)
    erreur = np.abs(Z_test - Z_predict)
    return Xtest,Ytest,Z_predict,erreur


def uniform_grid(dimensions,npts):
    Lx, Ly, Nx, Ny, Ox, Oy = dimensions
    N = int(np.sqrt(npts)) 
    x,y = np.linspace(Ox,Ox+Lx,N), np.linspace(Oy,Oy+Ly,N)
    X,Y = np.meshgrid(x,y)
    while X.size != npts:
        ind = np.random.randint(0,X.size)
        X = np.delete(X,ind)
        Y = np.delete(Y,ind)
    return X.flatten(),Y.flatten()

def data_to_grid(X,Y):
    grid = []
    for i in range(X.size):
        x,y = X[i],Y[i]
        grid.append(Cell(x,y,geometry))
    return grid


#%% 

def f(X,Y):
    return np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2)


geometry = [10,10,50,50,0,0]
grid = init_grid(geometry)
X,Y = coordinate(grid)

plt.scatter(X,Y,c=f(X,Y),cmap='jet',s=1)
plt.axis('square')
plt.colorbar()

X_test,Y_test,Z_test,erreur = apprentissage(grid)


plt.figure()
plt.scatter(X_test,Y_test,c=Z_test,cmap='jet',s=1)
plt.axis('square')
plt.colorbar()

print(np.linalg.norm(erreur))

#%%

grid = iterate_grid(grid)
X2,Y2 = coordinate(grid)

plt.figure()
plt.scatter(X2,Y2,c=f(X2,Y2),cmap='jet',s=1)
plt.axis('square')
plt.colorbar()

X_test,Y_test,Z_test,erreur = apprentissage(grid)

plt.figure()
plt.scatter(X_test,Y_test,c=Z_test,cmap='jet',s=1)
plt.axis('square')
plt.colorbar()

print(np.linalg.norm(erreur))


