# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:05:20 2022

@author: pchauris
"""

"Par rapport a la version raffinement.py, corrige la définition des surrogates models et ajoute distinctement le calcul de dTdx et dT*dz"
"Changement du critère de raffinement"

import numpy as np
import matplotlib.pyplot as plt

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
        C00 = Cell(np.concatenate((self.index,[0,0])),self.geometry)
        C00.size = self.size/2
        C00.center = self.center - np.array([px/4,py/4])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),self.geometry)
        C01.size = self.size/2
        C01.center = self.center + np.array([px/4,-py/4])
        
        C10 = Cell(np.concatenate((self.index,[1,0])),self.geometry)
        C10.size = self.size/2
        C10.center = self.center + np.array([-px/4,py/4])
        
        C11 = Cell(np.concatenate((self.index,[1,1])),self.geometry)
        C11.size = self.size/2
        C11.center = self.center + np.array([px/4,py/4])
        
        return C00,C01,C10,C11
    
    def split_x(self):
        px,py = self.size
        #create 2 new cells allong first axis
        C00 = Cell(np.concatenate((self.index,[0,0])),self.geometry)
        C00.size = np.array([px/2,py])
        C00.center = self.center - np.array([px/4,0])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),self.geometry)
        C01.size = np.array([px/2,py])
        C01.center = self.center + np.array([px/4,0])
        
        return C00,C01
    
    
    def split_y(self):
        px,py = self.size
        #create 2 new cells allong second axis
        C00 = Cell(np.concatenate((self.index,[0,0])),self.geometry)
        C00.size = np.array([px,py/2])
        C00.center = self.center - np.array([0,py/4])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),self.geometry)
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
    return np.array([np.array(X1),np.array(X2)])

def plot_cell(cell):
    x,y = cell.center
    px,py = cell.size
    plt.vlines(x = x-px/2, ymin = y-py/2, ymax = y+py/2,linewidth=1)
    plt.vlines(x = x+px/2, ymin = y-py/2, ymax = y+py/2,linewidth=1)
    plt.hlines(y = y-py/2, xmin = x-px/2, xmax = x+px/2,linewidth=1)
    plt.hlines(y = y+py/2, xmin = x-px/2, xmax = x+px/2,linewidth=1)
    

def crit_sequence(grid,Z,critere):
    "calcule la liste des critères sur la grille avec le citère donné"
    sequence = []
    for cell in grid:
        crit = critere(grid,Z)
        sequence.append(crit)
    return np.array(sequence)


def alpha_sequence(grid,Z,critere):
    sequence = crit_sequence(grid, Z, critere)
    return np.linspace(0,sequence.max(),len(grid))


def distrib_sequence(grid,Z,critere):
    alpha = alpha_sequence(grid,Z,critere)
    crit = crit_sequence(grid,Z,critere)
    distribution = []
    for j in range(alpha.size):
        dj = np.count_nonzero(crit>alpha[j])
        distribution.append(dj)
    return np.array(distribution)


def auto_threshold(grid,Z,critere):
    alpha = alpha_sequence(grid,Z,critere)
    distribution = distrib_sequence(grid,Z,critere)
    f = alpha*distribution
    #maximum global
    fmax = np.max(f)
    idmax = np.where(f == fmax)
    idmax = idmax[0][0]
    #alpha 
    alphamax = alpha[idmax]
    return alphamax

def iterate_grid(grid,Z,critere):
    alpha = auto_threshold(grid,Z,critere)
    crit = crit_sequence(grid,Z,critere)
    new_grid = grid.copy()
    for cell in grid:
        k = grid.index(cell)        
        # raffinement iso
        if crit[k] > alpha :
            C00,C01,C10,C11 = cell.split_iso()
            new_grid.remove(cell)
            new_grid.insert(k,C11)
            new_grid.insert(k,C10)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
        
    return new_grid

def split_grid(grid,Z,nx,ny,ix,iy):
    "calcule le masque correspondant au surrogate ix iy et retourne les données du surrogate"
    X,Y = coordinate(grid)
    "Z de dimension 3"
    Z1,Z2,Z3 = Z[:,0],Z[:,1],Z[:,2]
    
    Lx,Ly = grid[0].geometry[0:2]
    Ox,Oy = grid[0].geometry[4:6]
    Ax = (ix*Lx/nx+Ox)*np.ones(X.size)
    Bx = ((ix+1)*Lx/nx+Ox)*np.ones(X.size)
    Ay = (iy*Ly/ny+Oy)*np.ones(Y.size)
    By = ((iy+1)*Ly/ny+Oy)*np.ones(Y.size)
    
    mask = (np.less(Ax,X) & np.less(X,Bx)) & (np.less(Ay,Y) & np.less(Y,By))
    mask = ~mask
    subX = np.ma.masked_array(X,mask).compressed()
    subY = np.ma.masked_array(Y,mask).compressed()
    subZ1 = np.ma.masked_array(Z1,mask).compressed()
    subZ2 = np.ma.masked_array(Z2,mask).compressed()
    subZ3 = np.ma.masked_array(Z3,mask).compressed()
    
    return np.stack((subX,subY),-1),np.stack((subZ1,subZ2,subZ3),-1)
    
    
def surr_model_direct(X,Y,Z):
    "return the array of coefficients of the polynomial model P such that Z = P(X,Y)"
    # degré 4
    A = np.stack((X**4,Y*X**3,(X*Y)**2,X*Y**3,Y**4,X*X*X,X*X*Y,X*Y*Y,Y*Y*Y,X*X,X*Y,Y*Y,X,Y,np.ones(X.size)),-1)
    # resolution du système
    A_star = np.linalg.pinv(A)
    param = np.dot(A_star,Z)
    return param

def surr_model_inverse(X,Z1,Z2,Z3):
    "return the array of coefficients of the polynomial model P such that X = P(Z1,Z2,Z3)"
    # degré 3
    A = np.stack((Z1**3,Z2**3,Z3**3,Z1**2,Z2**2,Z3**2,Z1*Z2,Z1*Z3,Z2*Z3,Z1,Z2,Z3,np.ones(Z1.size),Z1*Z2*Z3),-1)
    # resolution du système
    A_star = np.linalg.pinv(A)
    param = np.dot(A_star,X)
    return param

def coeffs_direct(grid,Z,axe,nx,ny):
    "return the list of coefficients of all the nx time ny surrogate models of the grid for Z[axe] = P(grid)"
    Coeffs = []
    for iy in range(ny):
        for ix in range(nx):
            sub_grid,sub_Z = split_grid(grid,Z,nx,ny,ix,iy)
            param = surr_model_direct(sub_grid[:,0],sub_grid[:,1],sub_Z[:,axe])
            Coeffs.append(param)
    return Coeffs

def coeffs_inverse(grid,Z,axe,nx,ny):
    "return the list of coefficients of all the nx time ny surrogate models of the grid for grid[axe] = P(Z)"
    Coeffs = []
    for iy in range(ny):
        for ix in range(nx):
            sub_grid,sub_Z = split_grid(grid,Z,nx,ny,ix,iy)
            param = surr_model_inverse(sub_grid[:,axe],sub_Z[:,0],sub_Z[:,1],sub_Z[:,2])
            Coeffs.append(param)
    return Coeffs


def dTdx_surrogate(grid,Z,axe):
    "compute the gradient of the cell from the surrogate model where the cell is located"
    # nombre de surrogate models optimal pour des surrogates models de degré 4
    nx = int(np.sqrt(len(grid))/5)
    ny = int(np.sqrt(len(grid))/5)
    Coeffs = coeffs_direct(grid,Z,axe,nx,ny)
    dTdx = []
    Lx,Ly = grid[0].geometry[0:2]
    Ox,Oy = grid[0].geometry[4:6]
    for cell in grid:
        x,y = cell.center
        ix,iy = int((x-Ox)/(Lx/nx)), int((y-Oy)/(Ly/ny))
        # degré 4
        [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15] = Coeffs[iy*nx+ix]
        gx = 4*a1*x**3 + 3*a2*x**2*y + 2*a3*x*y**2 + a4*y**3 + 3*a6*x**2 + 2*a7*x*y + a8*y**2 + 2*a10*x + a11*y + a13
        gy = a2*x**3 + 2*a3*x**2*y + 3*a4*x*y**2 + 4*a5*y**3 + a7*x**2 + 2*a8*x*y + 3*a9*y**2 + a11*x + 2*a12*y + a14
        dTdx.append([gx,gy])
    return np.array(dTdx)

def dTdz_surrogate(grid,Z,axe):
    "calcule dT1*_dz(T(x)) ou dT1*_dz(T(x)) dépendant de Coeffs avec Coeffs calculés pour X1 = P(Z1,Z2,Z3) ou X2 = P(Z1,Z2,Z3)"
    # nombre de surrogate models optimal pour des surrogates models de degré 4
    nx = int(np.sqrt(len(grid))/5)
    ny = int(np.sqrt(len(grid))/5)
    Coeffs = coeffs_inverse(grid,Z,axe,nx,ny)
    dTdz =  []
    Lx,Ly = grid[0].geometry[0:2]
    Ox,Oy = grid[0].geometry[4:6]
    ind = 0
    for x,y in coordinate(grid).T:
        z1,z2,z3 = Z[ind,:]
        ind += 1
        ix,iy = int((x-Ox)/(Lx/nx)), int((y-Oy)/(Ly/ny))
        # degré 3
        [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14] = Coeffs[iy*nx+ix]
        g1 = 3*a1*z1**2+2*a4*z1+a7*z2+a8*z3+a10+a14*z2*z3
        g2 = 3*a2*z2**2+2*a5*z2+a7*z1+a9*z3+a11+a14*z1*z3
        g3 = 3*a3*z3**2+2*a6*z3+a8*z1+a9*z2+a12+a14*z1*z2
        dTdz.append([g1,g2,g3])
    return np.array(dTdz)

def dTdz_inversion(dTdx):  
    dTdz = []
    for i in range(dTdx.shape[0]):
        dz = np.linalg.pinv(dTdx[i,:,:])
        dTdz.append(dz)
    dTdz = np.array(dTdz)
    return dTdz

#%% Calcul de dTdx

def T_direct(X,Y):
    s = X**2 + Y**2 + 0.1
    return s,X/s,Y/s

geometry = [2,2,100,100,-1,-1]
grid = init_grid(geometry)
X1,X2 = coordinate(grid)
Z1,Z2,Z3 = T_direct(X1,X2)
Z = np.stack((Z1,Z2,Z3),-1)

dT1dx = dTdx_surrogate(grid,Z,0)
dT2dx = dTdx_surrogate(grid,Z,1)
dT3dx = dTdx_surrogate(grid,Z,2)

dTdx = np.stack((dT1dx,dT2dx,dT3dx),-1)

titres_dTdx = [r'$\frac{dT_1}{dx_1}(x)$',r'$\frac{dT_1}{dx_2}(x)$',r'$\frac{dT_2}{dx_1}(x)$',
          r'$\frac{dT_2}{dx_2}(x)$',r'$\frac{dT_3}{dx_1}(x)$',r'$\frac{dT_3}{dx_2}(x)$']

titres_dTdz = [r'$\frac{dT_1*}{dz_1}(T(x))$',r'$\frac{dT_1*}{dz_2}(T(x))$',r'$\frac{dT_1*}{dz_3}(T(x))$',
          r'$\frac{dT_2*}{dz_1}(T(x))$',r'$\frac{dT_2*}{dz_2}(T(x))$',r'$\frac{dT_2*}{dz_3}(T(x))$']

fig,ax = plt.subplots(3,2,figsize=(12,20))
for i in range(2):
    pc1 = ax[0,i].scatter(X1,X2,c=dT1dx[:,i],cmap='jet')
    ax[0,i].axis('square')
    ax[0,i].set_xlabel(r'$x1$')
    ax[0,i].set_ylabel(r'$x2$')
    ax[0,i].set_title(titres_dTdx[i])
    fig.colorbar(pc1,ax=ax[0,i])
    
    pc2 = ax[1,i].scatter(X1,X2,c=dT2dx[:,i],cmap='jet')
    ax[1,i].axis('square')
    ax[1,i].set_xlabel(r'$x1$')
    ax[1,i].set_ylabel(r'$x2$')
    ax[1,i].set_title(titres_dTdx[i+2])
    fig.colorbar(pc2,ax=ax[1,i])
    
    pc3 = ax[2,i].scatter(X1,X2,c=dT3dx[:,i],cmap='jet')
    ax[2,i].axis('square')
    ax[2,i].set_xlabel(r'$x1$')
    ax[2,i].set_ylabel(r'$x2$')
    ax[2,i].set_title(titres_dTdx[i+4])
    fig.colorbar(pc3,ax=ax[2,i])

#%% calcul de dTdz par inversion de dTdx

dTdz = dTdz_inversion(dTdx)

fig,ax = plt.subplots(2,3,figsize=(20,12))

for i in range(3):
    pcm1 = ax[0,i].scatter(X1,X2,c=dTdz[:,i,0],cmap='jet')
    ax[0,i].axis('square')
    ax[0,i].set_xlabel(r'$x1$')
    ax[0,i].set_ylabel(r'$x2$')
    ax[0,i].set_title(titres_dTdz[i])
    fig.colorbar(pcm1,ax=ax[0,i])
    
    pcm1 = ax[1,i].scatter(X1,X2,c=dTdz[:,i,1],cmap='jet')
    ax[1,i].axis('square')
    ax[1,i].set_xlabel(r'$x1$')
    ax[1,i].set_ylabel(r'$x2$')
    ax[1,i].set_title(titres_dTdz[i+3])
    fig.colorbar(pcm1,ax=ax[1,i])
    
#%% calcul de dT*dz par surrogate models

dT1dz = dTdz_surrogate(grid,Z,0)
dT2dz = dTdz_surrogate(grid,Z,1)

dTdz = np.stack((dT1dz,dT2dz),-1)

fig,ax = plt.subplots(2,3,figsize=(20,12))

for i in range(3):
    pcm1 = ax[0,i].scatter(X1,X2,c=dT1dz[:,i],cmap='jet')
    ax[0,i].axis('square')
    ax[0,i].set_xlabel(r'$x1$')
    ax[0,i].set_ylabel(r'$x2$')
    ax[0,i].set_title(titres_dTdz[i])
    fig.colorbar(pcm1,ax=ax[0,i])
    
    pcm1 = ax[1,i].scatter(X1,X2,c=dT2dz[:,i],cmap='jet')
    ax[1,i].axis('square')
    ax[1,i].set_xlabel(r'$x1$')
    ax[1,i].set_ylabel(r'$x2$')
    ax[1,i].set_title(titres_dTdz[i+3])
    fig.colorbar(pcm1,ax=ax[1,i])

#%% calcul du critère

crit = []
for i in range(len(grid)):
    crit.append(np.dot(dTdx[0,:,:],dTdz[0,:,:])-np.eye(2))
crit = np.array(crit)

fig,ax = plt.subplots(2,2,figsize=(20,15))

for i in range(2):
    pcm1 = ax[0,i].scatter(X1,X2,c=crit[:,0,i],cmap='jet')
    ax[0,i].axis('square')
    ax[0,i].set_xlabel(r'$x1$')
    ax[0,i].set_ylabel(r'$x2$')
    fig.colorbar(pcm1,ax=ax[0,i])
    
    pcm1 = ax[1,i].scatter(X1,X2,c=crit[:,1,i],cmap='jet')
    ax[1,i].axis('square')
    ax[1,i].set_xlabel(r'$x1$')
    ax[1,i].set_ylabel(r'$x2$')
    fig.colorbar(pcm1,ax=ax[1,i])
    
    # pcm1 = ax[2,i].scatter(X1,X2,c=crit[:,2,i],cmap='jet')
    # ax[2,i].axis('square')
    # ax[2,i].set_xlabel(r'$x1$')
    # ax[2,i].set_ylabel(r'$x2$')
    # fig.colorbar(pcm1,ax=ax[2,i])
    
    