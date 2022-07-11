# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:11:27 2022

@author: pchauris
"""

import numpy as np
import matplotlib.pyplot as plt
#%% 
def f(X,Y):
    return np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2)
    return Y*np.sin(X)


#%% fonctions
class Cell:
    def __init__(self,index,geometry):
        
        Lx, Ly, Nx, Ny, Ox, Oy = geometry
        px,py = Lx/Nx,Ly/Ny
        x = index[0]*px + px/2 + Ox
        y = index[1]*py + py/2 + Oy
        
        self.index = index
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
        C00 = Cell(np.concatenate((self.index,[0,0])),geometry)
        C00.size = self.size/2
        C00.center = self.center - np.array([px/4,py/4])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),geometry)
        C01.size = self.size/2
        C01.center = self.center + np.array([px/4,-py/4])
        
        C10 = Cell(np.concatenate((self.index,[1,0])),geometry)
        C10.size = self.size/2
        C10.center = self.center + np.array([-px/4,py/4])
        
        C11 = Cell(np.concatenate((self.index,[1,1])),geometry)
        C11.size = self.size/2
        C11.center = self.center + np.array([px/4,py/4])
        
        return C00,C01,C10,C11
    
    def split_x(self):
        px,py = self.size
        #create 2 new cells allong first axis
        C00 = Cell(np.concatenate((self.index,[0,0])),geometry)
        C00.size = np.array([px/2,py])
        C00.center = self.center - np.array([px/4,0])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),geometry)
        C01.size = np.array([px/2,py])
        C01.center = self.center + np.array([px/4,0])
        
        return C00,C01
    
    
    def split_y(self):
        px,py = self.size
        #create 2 new cells allong second axis
        C00 = Cell(np.concatenate((self.index,[0,0])),geometry)
        C00.size = np.array([px,py/2])
        C00.center = self.center - np.array([0,py/4])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),geometry)
        C01.size = np.array([px,py/2])
        C01.center = self.center + np.array([0,py/4])
        
        return C00,C01

def init_grid(geometry):
    grid = []
    Nx,Ny = geometry[2:4]
    for i in range(Nx):
        for j in range(Ny):
            cell = Cell(np.array([i,j]),geometry)
            grid.append(cell)
    return grid


def coordinate(grid):
    X1 = []
    X2 = []
    for cell in grid:
        x1,x2 = cell.center
        X1.append(x1)
        X2.append(x2)
    return np.array(X1),np.array(X2)


def emp_grad(cell):
    x1,y1 = cell.center
    dx,dy = cell.size
    x0,y0 = x1 - dx/2 , y1 - dy/2
    Dfx = f(x0+dx,y0) - f(x0,y0)
    Dfy = f(x0,y0+dy) - f(x0,y0)
    return np.abs(Dfx/dx),np.abs(Dfy/dy)


def crit_sequence(grid):
    iso,vert,horiz = [],[],[]
    for cell in grid:
        gx,gy = emp_grad(cell)
        iso.append(np.sqrt(gx**2+gy**2))
        vert.append(gx)
        horiz.append(gy)
    return np.array(iso),np.array(vert),np.array(horiz)

def alpha_sequence(grid):
    res1,res2,res3 = crit_sequence(grid)
    return np.linspace(0,res1.mean(),len(grid)),np.linspace(0,res2.mean(),len(grid)),np.linspace(0,res3.mean(),len(grid))

def distrib_sequence(grid):
    alpha1,alpha2,alpha3 = alpha_sequence(grid)
    crit1,crit2,crit3 = crit_sequence(grid)
    res1,res2,res3 = [],[],[]
    for j in range(alpha1.size):
        dj1 = np.count_nonzero(crit1>alpha1[j])
        dj2 = np.count_nonzero(crit2>alpha2[j])
        dj3 = np.count_nonzero(crit3>alpha3[j])
        res1.append(dj1)
        res2.append(dj2)
        res3.append(dj3)
    return np.array(res1),np.array(res2),np.array(res3)

def auto_threshold(grid):
    alpha1,alpha2,alpha3 = alpha_sequence(grid)
    d1,d2,d3 = distrib_sequence(grid)
    f1,f2,f3 = alpha1*d1,alpha2*d2,alpha3*d3
    #maximum global
    fmax1,fmax2,fmax3 = np.max(f1), np.max(f2), np.max(f3)
    idmax1,idmax2,idmax3 = np.where(f1 == fmax1),np.where(f2 == fmax2),np.where(f3 == fmax3)
    idmax1,idmax2,idmax3 = idmax1[0][0],idmax2[0][0],idmax3[0][0]
    #alpha 
    alphamax1,alphamax2,alphamax3 = alpha1[idmax1], alpha2[idmax2], alpha3[idmax3]
    return alphamax1,alphamax2,alphamax3

def iterate_grid(grid):
    alpha1,alpha2,alpha3 = auto_threshold(grid)
    new_grid = grid.copy()
    for cell in grid:
        k = grid.index(cell)
        gx,gy = emp_grad(cell)
        if np.sqrt(gx**2+gy**2) > alpha1:
            C00,C01,C10,C11 = cell.split_iso()
            new_grid.remove(cell)
            new_grid.insert(k,C11)
            new_grid.insert(k,C10)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
        elif gx > alpha2:
            C00,C01 = cell.split_x()
            if cell in new_grid:
                new_grid.remove(cell)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
        elif gy > alpha3:
            C00,C01 = cell.split_y()
            if cell in new_grid:
                new_grid.remove(cell)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
        
    A1.append(alpha1),A2.append(alpha2),A3.append(alpha3)              
    return new_grid

#%% script
geometry = [2*np.pi,1,50,50,0,0]
grid = init_grid(geometry)
X1,X2 = coordinate(grid)
plt.scatter(X1,X2,c = f(X1,X2),s = 1,cmap = 'jet')
plt.colorbar()


new_grid = grid.copy()
for cell in grid:
    k = grid.index(cell)
    C00,C01,= cell.split_x()
    new_grid.remove(cell)

    new_grid.insert(k,C01)
    new_grid.insert(k,C00)
    
X1,X2 = coordinate(new_grid)
plt.figure()
plt.scatter(X1,X2,c = f(X1,X2),s = 1,cmap = 'jet')
plt.colorbar()


grad = []
for cell in new_grid:
    gx,gy = emp_grad(cell)
    grad.append(np.sqrt(gx**2+gy**2))
    
plt.figure()
plt.scatter(X1,X2,c = grad,s = 1,cmap = 'jet')
plt.colorbar()


#%%
geometry = [10,10,11,11,0,0]
grid = init_grid(geometry)

A1,A2,A3 = [],[],[]
niter = 3
for _ in range(niter):
    X1,X2 = coordinate(grid)
    print('taille de la grille :',len(grid))
    grid = iterate_grid(grid)
    
print('taille de la grille :',len(grid))
X1,X2 = coordinate(grid)
plt.figure()
plt.scatter(X1,X2,c = f(X1,X2),s = 1,cmap = 'jet')
plt.colorbar()

#%%
plt.plot(A1,'-o')
plt.plot(A2,'-o')
plt.plot(A3,'-o')
plt.legend(['norme','x axis','y axis'])