# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:34:45 2022

@author: pchauris

Objectif : Tester le raffinement sur T* en raffinant la grille X a partir du critère dT*_dz(z=T(x)) 
Vérifier son influence sur l'apprentissage de T*

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from findiff import FinDiff

#%%
def T_direct(X,Y):
    s = X**2 + Y**2 + 0.001
    return s,X/s,Y/s

def T_inverse(Z1,Z2,Z3):
    return np.array([Z1*Z2,Z1*Z3]).T


def dT_dz(x1,x2):
    # calcul du z correspondant
    z1,z2,z3 = T_direct(x1,x2)
    
    # calcul de dT*_dz(z)
    """"
    Il y a plusieurs manières de définir le gradient de T* tout comme il y a plusieurs moyens de définir T*
        - par la formule analytique pour un T* choisi 
        - par une inverse à gauche sur dT_dx
        - par une approximation numérique : surrogate models
    Dans un premier temps, testons avec l'expression analytique pour T* = z1z2,z1z3
    """
    return np.array([z2,z1,np.zeros(z1.size),z3,np.zeros(z1.size),z1]).T

def dT_dx(x1,x2):
    return np.array([[2*x1,2*x2],
                     [(0.001+x2**2-x1**2)/(x1**2+x2**2+0.001)**2,-2*x1*x2/(x1**2+x2**2+0.001)**2],
                     [-2*x1*x2/(x1**2+x2**2+0.001)**2,(0.001+x1**2-x2**2)/(x1**2+x2**2+0.001)**2]])

def dT_dz2(x1,x2):
    "methode par inversion"
    dTdx = dT_dx(x1,x2)
    dTdz = []
    for i in range(dTdx.shape[2]):
        dz = np.linalg.pinv(dTdx[:,:,i])
        dTdz.append(dz)
    dTdz = np.array(dTdz)
    return dTdz

#%% code de raffinement modifié

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
    

def crit_sequence(grid,Z,nx,ny,Coeffs):
    grad = gradient(grid,Z,nx,ny,Coeffs)
    g1 = grad[:,0]
    g2 = grad[:,1]
    g3 = grad[:,2]
    return np.sqrt(g1**2+g2**2+g3**2)

def alpha_sequence(grid,Z,nx,ny,Coeffs):
    res = crit_sequence(grid,Z,nx,ny,Coeffs)
    return np.linspace(0,res.max(),len(grid))


def distrib_sequence(grid,Z,nx,ny,Coeffs):
    alpha = alpha_sequence(grid,Z,nx,ny,Coeffs)
    crit = crit_sequence(grid,Z,nx,ny,Coeffs)
    res = []
    for j in range(alpha.size):
        dj = np.count_nonzero(crit>alpha[j])
        res.append(dj)

    return np.array(res)

def auto_threshold(grid,Z,nx,ny,Coeffs):
    alpha = alpha_sequence(grid,Z,nx,ny,Coeffs)
    d = distrib_sequence(grid,Z,nx,ny,Coeffs)
    f = alpha*d
    #maximum global
    fmax = np.max(f)
    idmax = np.where(f == fmax)
    idmax = idmax[0][0]
    #alpha 
    alphamax = alpha[idmax]
    return alphamax


def iterate_grid(grid,Z,axe):
    
    nx,ny = int(np.sqrt(len(grid))/5),int(np.sqrt(len(grid))/5)
    Coeffs = coeffs(grid,Z,axe,nx,ny)
    
    alpha = auto_threshold(grid,Z,nx,ny,Coeffs)
    new_grid = grid.copy()
    crit_seq = crit_sequence(grid,Z,nx,ny,Coeffs)
    for cell in grid:
        k = grid.index(cell)
        # raffinement iso
        if crit_seq[k] > alpha :
            C00,C01,C10,C11 = cell.split_iso()
            if cell in new_grid:
                new_grid.remove(cell)
            new_grid.insert(k,C11)
            new_grid.insert(k,C10)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
    
    return new_grid

def split_grid(grid,Z,nx,ny,ix,iy):
    "calcule le masque correspondant au surrogate ix iy et retourne les données du surrogate"
    X,Y = coordinate(grid)
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


def surr_model(X,Z,axe):
    "return the array of coefficients of the polynomial model P such that X[axe] = P(Z)"
    X = X[:,axe]
    Z1,Z2,Z3 = Z[:,0],Z[:,1],Z[:,2]
    # degré 1
    # A = np.stack((Z1,Z2,Z3,np.ones(Z1.shape)),-1)
    # degré 3
    A = np.stack((Z1**3,Z2**3,Z3**3,Z1**2,Z2**2,Z3**2,Z1*Z2,Z1*Z3,Z2*Z3,Z1,Z2,Z3,np.ones(Z1.size),Z1*Z2*Z3),-1)
    # resolution du système
    A_star = np.linalg.pinv(A)
    param = np.dot(A_star,X)
    return param


def coeffs(grid,Z,axe,nx,ny):
    "return the list of coefficients of all the nx time ny surrogate models of the grid"
    Coeffs = []
    for iy in range(ny):
        for ix in range(nx):
            sub_grid,sub_Z = split_grid(grid,Z,nx,ny,ix,iy)
            param = surr_model(sub_grid,sub_Z,axe)
            Coeffs.append(param)
    return Coeffs


def gradient(grid,Z,nx,ny,Coeffs):
    "calcule dT1*_dz(T(x)) ou dT1*_dz(T(x)) dépendant de Coeffs avec Coeffs calculés pour X1 = P(Z1,Z2,Z3) ou X2 = P(Z1,Z2,Z3)"
    grad =  []
    Lx,Ly = grid[0].geometry[0:2]
    Ox,Oy = grid[0].geometry[4:6]
    ind = 0
    for x,y in coordinate(grid).T:
        z1,z2,z3 = Z[ind,:]
        ind += 1
        ix,iy = int((x-Ox)/(Lx/nx)), int((y-Oy)/(Ly/ny))
        #degré 1
        # a1,a2,a3,a4 = Coeffs[iy*nx+ix]
        # g1,g2,g3 = a1,a2,a3
        # degré 3
        [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14] = Coeffs[iy*nx+ix]
        g1 = 3*a1*z1**2+2*a4*z1+a7*z2+a8*z3+a10+a14*z2*z3
        g2 = 3*a2*z2**2+2*a5*z2+a7*z1+a9*z3+a11+a14*z1*z3
        g3 = 3*a3*z3**2+2*a6*z3+a8*z1+a9*z2+a12+a14*z1*z2
        # grad.append(np.sqrt(g1**2+g2**2+g3**2))
        grad.append([g1,g2,g3])
    return np.array(grad)

#%% gradient analytique

geometry = [2,2,40,40,-1,-1]
grid = init_grid(geometry)
X,Y = coordinate(grid)



z1,z2,z3 = T_direct(X,Y)

dZ = dT_dz(X,Y)

dZ1 = np.sqrt(dZ[:,0]**2 + dZ[:,1]**2 + dZ[:,2]**2)
dZ2 = np.sqrt(dZ[:,3]**2 + dZ[:,4]**2 + dZ[:,5]**2)

fig,(ax1,ax2) = plt.subplots(1,2)

pcm1 = ax1.scatter(X,Y,c=dZ1,cmap='jet')
ax1.axis('square')
fig.colorbar(pcm1,ax=ax1,shrink=0.6)

pcm2 = ax2.scatter(X,Y,c=dZ2,cmap='jet')
ax2.axis('square')
fig.colorbar(pcm2,ax=ax2,shrink=0.6)

titres_T = [r'$T_1(x)$',r'$T_2(x)$',r'$T_3(x)$']

titres_dTdx = [r'$\frac{dT_1}{dx_1}(x)$',r'$\frac{dT_1}{dx_2}(x)$',r'$\frac{dT_2}{dx_1}(x)$',
          r'$\frac{dT_2}{dx_2}(x)$',r'$\frac{dT_3}{dx_1}(x)$',r'$\frac{dT_3}{dx_2}(x)$']

titres_dTdz = [r'$\frac{dT_1*}{dz_1}(T(x))$',r'$\frac{dT_1*}{dz_2}(T(x))$',r'$\frac{dT_1*}{dz_3}(T(x))$',
          r'$\frac{dT_2*}{dz_1}(T(x))$',r'$\frac{dT_2*}{dz_2}(T(x))$',r'$\frac{dT_2*}{dz_3}(T(x))$']

fig,ax = plt.subplots(2,3,figsize=(20,12))
for i in range(6):
    pcm = ax[i//3,i%3].scatter(X,Y,c=dZ[:,i],cmap='jet')
    ax[i//3,i%3].axis('square')
    ax[i//3,i%3].set_xlabel(r'$x1$')
    ax[i//3,i%3].set_ylabel(r'$x2$')
    ax[i//3,i%3].set_title(titres_dTdz[i])
    fig.colorbar(pcm,ax=ax[i//3,i%3])

#%% données de KKL

import pandas as pd
geometry = [2,2,50,50,-1,-1]
grid = init_grid(geometry)
X,Y = coordinate(grid)

data = pd.read_excel('donnees.xlsx').to_numpy()
im,ax = plt.subplots(1,3,figsize=(15,5))
for i in range(3):
    pcm = ax[i].scatter(data[:,0],data[:,1],c = data[:,2+i],cmap='jet')
    ax[i].set_xlabel(r'$x1$')
    ax[i].set_ylabel(r'$x2$')
    ax[i].set_title(titres_T[i])
    ax[i].axis('square')
    im.colorbar(pcm,ax=ax[i])
plt.show()

nx,ny = 9,9
Z = data[:,2:5]

Coeff1 = coeffs(grid,Z,0,nx,ny)
print('coeff1')
Coeff2 = coeffs(grid,Z,1,nx,ny)
print('coeff2')
grad1 = gradient(grid,Z,nx,ny,Coeff1)
print('grad1')
grad2 = gradient(grid,Z,nx,ny,Coeff2)
print('grad2')

fig,ax = plt.subplots(2,3,figsize=(20,12))

for i in range(3):
    pcm1 = ax[0,i].scatter(data[:,0],data[:,1],c=grad1[:,i],cmap='jet')
    ax[0,i].set_xlabel(r'$x1$')
    ax[0,i].set_ylabel(r'$x2$')
    ax[0,i].set_title(titres_dTdz[i])
    fig.colorbar(pcm1,ax=ax[0,i])
    
    pcm1 = ax[1,i].scatter(data[:,0],data[:,1],c=grad2[:,i],cmap='jet')
    ax[1,i].axis('square')
    ax[1,i].set_xlabel(r'$x1$')
    ax[1,i].set_ylabel(r'$x2$')
    ax[1,i].set_title(titres_dTdz[i+3])
    fig.colorbar(pcm1,ax=ax[1,i])
    
#%% calcul de dT_dx par surrogates models puis de dT*_dz par inversion

def split_grid2(grid,Zi,nx,ny,ix,iy):
    "calcule le masque correspondant au surrogate ix iy et retourne les données du surrogate"
    X,Y = coordinate(grid)
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
    subZi = np.ma.masked_array(Zi,mask).compressed()

    return np.stack((subX,subY),-1),subZi


def surr_model2(sub_grid,sub_Zi):
    "return the array of coefficients of the polynomial model fit over sub_grid"
    X,Y = sub_grid[:,0],sub_grid[:,1]
    # degré 4
    A = np.stack((X**4,Y*X**3,(X*Y)**2,X*Y**3,Y**4,X*X*X,X*X*Y,X*Y*Y,Y*Y*Y,X*X,X*Y,Y*Y,X,Y,np.ones(X.size)),-1)
    # resolution du système
    A_star = np.linalg.pinv(A)
    param = np.dot(A_star,sub_Zi)
    return param


def coeffs2(grid,Z,nx,ny):
    "return the list of coefficients of all the nx time ny surrogate models of the grid"
    Coeffs = []
    for iy in range(ny):
        for ix in range(nx):
            sub_grid,sub_Z = split_grid2(grid,Z,nx,ny,ix,iy)
            param = surr_model2(sub_grid,sub_Z)
            Coeffs.append(param)
    return Coeffs


def gradient2(cell,nx,ny,Coeffs):
    "compute the gradient of the cell from the surrogate model where the cell is located"
    x,y = cell.center
    Lx,Ly = cell.geometry[0:2]
    Ox,Oy = cell.geometry[4:6]
    ix,iy = int((x-Ox)/(Lx/nx)), int((y-Oy)/(Ly/ny))
    # degré 4
    [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15] = Coeffs[iy*nx+ix]
    gx = 4*a1*x**3 + 3*a2*x**2*y + 2*a3*x*y**2 + a4*y**3 + 3*a6*x**2 + 2*a7*x*y + a8*y**2 + 2*a10*x + a11*y + a13
    gy = a2*x**3 + 2*a3*x**2*y + 3*a4*x*y**2 + 4*a5*y**3 + a7*x**2 + 2*a8*x*y + 3*a9*y**2 + a11*x + 2*a12*y + a14
    return np.array([gx,gy])

geometry = [2,2,50,50,-1,-1]
grid = init_grid(geometry)
X,Y = coordinate(grid)
nx,ny = 10,10
Coeff1 = coeffs2(grid,data[:,2],nx,ny)
Coeff2 = coeffs2(grid,data[:,3],nx,ny)
Coeff3 = coeffs2(grid,data[:,4],nx,ny)
dTdx1,dTdx2,dTdx3 = [],[],[]
for cell in grid:
    dTdx1.append(gradient2(cell,nx,ny,Coeff1))
    dTdx2.append(gradient2(cell,nx,ny,Coeff2))
    dTdx3.append(gradient2(cell,nx,ny,Coeff3))
    
dTdx1,dTdx2,dTdx3 = np.array(dTdx1),np.array(dTdx2),np.array(dTdx3)

fig,ax = plt.subplots(3,2,figsize=(12,20))
for i in range(2):
    pc1 = ax[0,i].scatter(X,Y,c=dTdx1[:,i],cmap='jet')
    ax[0,i].axis('square')
    ax[0,i].set_xlabel(r'$x1$')
    ax[0,i].set_ylabel(r'$x2$')
    ax[0,i].set_title(titres_dTdx[i])
    fig.colorbar(pc1,ax=ax[0,i])
    
    pc2 = ax[1,i].scatter(X,Y,c=dTdx2[:,i],cmap='jet')
    ax[1,i].axis('square')
    ax[1,i].set_xlabel(r'$x1$')
    ax[1,i].set_ylabel(r'$x2$')
    ax[1,i].set_title(titres_dTdx[i+2])
    fig.colorbar(pc2,ax=ax[1,i])
    
    pc3 = ax[2,i].scatter(X,Y,c=dTdx3[:,i],cmap='jet')
    ax[2,i].axis('square')
    ax[2,i].set_xlabel(r'$x1$')
    ax[2,i].set_ylabel(r'$x2$')
    ax[2,i].set_title(titres_dTdx[i+4])
    fig.colorbar(pc3,ax=ax[2,i])

dTdx = np.stack((dTdx1,dTdx2,dTdx3),-1)

dTdz = []
for i in range(dTdx.shape[0]):
    dz = np.linalg.pinv(dTdx[i,:,:])
    dTdz.append(dz)
dTdz = np.array(dTdz)

fig,ax = plt.subplots(2,3,figsize=(20,12))

for i in range(3):
    pcm1 = ax[0,i].scatter(X,Y,c=dTdz[:,i,0],cmap='jet')
    ax[0,i].axis('square')
    ax[0,i].set_xlabel(r'$x1$')
    ax[0,i].set_ylabel(r'$x2$')
    ax[0,i].set_title(titres_dTdz[i])
    fig.colorbar(pcm1,ax=ax[0,i])
    
    pcm1 = ax[1,i].scatter(X,Y,c=dTdz[:,i,1],cmap='jet')
    ax[1,i].axis('square')
    ax[1,i].set_xlabel(r'$x1$')
    ax[1,i].set_ylabel(r'$x2$')
    ax[1,i].set_title(titres_dTdz[i+3])
    fig.colorbar(pcm1,ax=ax[1,i])
    
#%% calcul de dT*_dz par surrogate models
geometry = [2,2,40,40,-1,-1]
grid = init_grid(geometry)
X1,X2 = coordinate(grid)
Z1,Z2,Z3 = T_direct(X1,X2)
Z = np.stack((Z1,Z2,Z3),-1)

nx,ny = 5,5

Coeff1 = coeffs(grid,Z,0,nx,ny)
print('coeff1')
Coeff2 = coeffs(grid,Z,1,nx,ny)
print('coeff2')

grad1 = gradient(grid,Z,nx,ny,Coeff1)
print('gradient1')
grad2 = gradient(grid,Z,nx,ny,Coeff2)
print('gradient2')

dTdz_surrogate = np.stack((grad1,grad2),-2)

fig,(ax1,ax2) = plt.subplots(1,2)

pcm1 = ax1.scatter(X1,X2,c=np.sqrt(grad1[:,0]**2+grad1[:,1]**2+grad1[:,2]**2),cmap='jet')
ax1.axis('square')
fig.colorbar(pcm1,ax=ax1,shrink=0.6)

pcm2 = ax2.scatter(X1,X2,c=np.sqrt(grad2[:,0]**2+grad2[:,1]**2+grad2[:,2]**2),cmap='jet')
ax2.axis('square')
fig.colorbar(pcm2,ax=ax2,shrink=0.6)


fig,ax = plt.subplots(2,3,figsize=(20,12))

for i in range(3):
    pcm1 = ax[0,i].scatter(X1,X2,c=dTdz_surrogate[:,0,i],cmap='jet')
    ax[0,i].axis('square')
    ax[0,i].set_xlabel(r'$x1$')
    ax[0,i].set_ylabel(r'$x2$')
    ax[0,i].set_title(titres_dTdz[i])
    fig.colorbar(pcm1,ax=ax[0,i])
    
    pcm1 = ax[1,i].scatter(X1,X2,c=dTdz_surrogate[:,1,i],cmap='jet')
    ax[1,i].axis('square')
    ax[1,i].set_xlabel(r'$x1$')
    ax[1,i].set_ylabel(r'$x2$')
    ax[1,i].set_title(titres_dTdz[i+3])
    fig.colorbar(pcm1,ax=ax[1,i])
    
#%% taille optimale des surrogate models

erreur = []
Nx = np.arange(2,8)
for nx in Nx:
    Coeff1 = coeffs(grid,Z,0,nx,nx)
    Coeff2 = coeffs(grid,Z,1,nx,nx)
    grad1 = gradient(grid,Z,nx,nx,Coeff1)
    grad2 = gradient(grid,Z,nx,nx,Coeff2)
    norme1 = np.sqrt(grad1[:,0]**2+grad1[:,1]**2+grad1[:,2]**2)
    norme2 = np.sqrt(grad2[:,0]**2+grad2[:,1]**2+grad2[:,2]**2)
    erreur.append(np.linalg.norm(norme1-dZ1)+np.linalg.norm(norme2-dZ2))
    
plt.plot(Nx,erreur,'-o')
#%%
plt.figure()
plt.scatter(Z1,Z2,c=dTdz_surrogate[:,0,0],cmap='jet')
plt.colorbar()
#%% calcul de dT_dx

dTdx = dT_dx(X1,X2)

fig,ax = plt.subplots(3,2,figsize=(12,18))

for i in range(2):
    pcm1 = ax[0,i].scatter(X1,X2,c=dTdx[0,i,:],cmap='jet')
    ax[0,i].axis('square')
    ax[0,i].set_xlabel(r'$x1$')
    ax[0,i].set_ylabel(r'$x2$')
    ax[0,i].set_title(titres_dTdx[i])
    fig.colorbar(pcm1,ax=ax[0,i])
    
    pcm1 = ax[1,i].scatter(X1,X2,c=dTdx[1,i,:],cmap='jet')
    ax[1,i].axis('square')
    ax[1,i].set_xlabel(r'$x1$')
    ax[1,i].set_ylabel(r'$x2$')
    ax[1,i].set_title(titres_dTdx[i])
    fig.colorbar(pcm1,ax=ax[1,i])
    
    pcm1 = ax[2,i].scatter(X1,X2,c=dTdx[2,i,:],cmap='jet')
    ax[2,i].axis('square')
    ax[2,i].set_xlabel(r'$x1$')
    ax[2,i].set_ylabel(r'$x2$')
    ax[2,i].set_title(titres_dTdx[i])
    fig.colorbar(pcm1,ax=ax[2,i])

"test de dT*_dz_surrogate * dT_dx = Identité"
i = 23
print(np.dot(dTdz_surrogate[i,:,:],dTdx[:,:,i]))
#test correct
#%% calcul de dT*_dz par la formule d'inversion

dTdz_inversion = dT_dz2(X1,X2)

fig,(ax1,ax2) = plt.subplots(1,2)

pcm1 = ax1.scatter(X1,X2,c=np.sqrt(dTdz_inversion[:,0,0]**2+dTdz_inversion[:,0,1]**2+dTdz_inversion[:,0,2]**2),cmap='jet')
ax1.axis('square')
fig.colorbar(pcm1,ax=ax1,shrink=0.6)

pcm2 = ax2.scatter(X1,X2,c=np.sqrt(dTdz_inversion[:,1,0]**2+dTdz_inversion[:,1,1]**2+dTdz_inversion[:,1,2]**2),cmap='jet')
ax2.axis('square')
fig.colorbar(pcm2,ax=ax2,shrink=0.6)

fig,ax = plt.subplots(2,3,figsize=(20,12))

for i in range(3):
    pcm1 = ax[0,i].scatter(X1,X2,c=dTdz_inversion[:,0,i],cmap='jet')
    ax[0,i].axis('square')
    ax[0,i].axis('square')
    ax[0,i].set_xlabel(r'$x1$')
    ax[0,i].set_ylabel(r'$x2$')
    ax[0,i].set_title(titres_dTdz[i])
    fig.colorbar(pcm1,ax=ax[0,i])
    
    pcm1 = ax[1,i].scatter(X1,X2,c=dTdz_inversion[:,1,i],cmap='jet')
    ax[1,i].axis('square')
    ax[1,i].axis('square')
    ax[1,i].set_xlabel(r'$x1$')
    ax[1,i].set_ylabel(r'$x2$')
    ax[1,i].set_title(titres_dTdz[i+3])
    fig.colorbar(pcm1,ax=ax[1,i])
    
"test de dT*_dz_inversion * dT_dx = Identité"
i = 23
print(np.dot(dTdz_inversion[i,:,:],dTdx[:,:,i]))
#test correct
#%% test du calcul de gradient de l'inverse

geometry = [np.pi,np.pi,30,30,0,0]
grid = init_grid(geometry)
X1,X2 = coordinate(grid)
Z1 = X1*np.cos(X2)
Z2 = X1*np.sin(X2)
Z3 = X2/X1
Z = np.stack((Z1,Z2,Z3),-1)

nx,ny = 6,6
Coeff1 = coeffs(grid,Z,0,nx,ny)
Coeff2 = coeffs(grid,Z,1,nx,ny)

grad1 = gradient(grid,Z,nx,ny,Coeff1)
grad2 = gradient(grid,Z,nx,ny,Coeff2)

fig,(ax1,ax2) = plt.subplots(1,2)

pcm1 = ax1.scatter(X1,X2,c=np.sqrt(grad1[:,0]**2+grad1[:,1]**2+grad1[:,2]**2),cmap='jet')
ax1.axis('square')
fig.colorbar(pcm1,ax=ax1,shrink=0.6)

pcm2 = ax2.scatter(X1,X2,c=np.sqrt(grad2[:,0]**2+grad2[:,1]**2+grad2[:,2]**2),cmap='jet')
ax2.axis('square')
fig.colorbar(pcm2,ax=ax2,shrink=0.6)

fig,ax = plt.subplots(2,3,figsize=(20,10))
for i in range(3):
    pcm1 = ax[0,i].scatter(X1,X2,c=grad1[:,i],cmap='jet')
    ax[0,i].axis('square')
    fig.colorbar(pcm1,ax=ax[0,i])
    
    pcm1 = ax[1,i].scatter(X1,X2,c=grad2[:,i],cmap='jet')
    ax[1,i].axis('square')
    fig.colorbar(pcm1,ax=ax[1,i])

#%% raffinement sur T1* et sur T2*
grid1 = iterate_grid(grid,Z,0)
# grid1 = iterate_grid(grid1,0)
# grid1 = iterate_grid(grid1,0)
X1,Y1 = coordinate(grid1)

grid2 = iterate_grid(grid,Z,1)
# grid2 = iterate_grid(grid2,1)
X2,Y2 = coordinate(grid2)


dZ1 = dT_dz(X1,Y1)

dZ2 = dT_dz(X2,Y2)

dZ1_norme = np.sqrt(dZ1[:,0]**2 + dZ1[:,1]**2 + dZ1[:,2]**2)
dZ2_norme = np.sqrt(dZ2[:,3]**2 + dZ2[:,4]**2 + dZ2[:,5]**2)

fig,(ax1,ax2) = plt.subplots(1,2)

pcm1 = ax1.scatter(X1,Y1,c=dZ1_norme,cmap='jet',s=1)
ax1.axis('square')
fig.colorbar(pcm1,ax=ax1,shrink=0.6)

pcm2 = ax2.scatter(X2,Y2,c=dZ2_norme,cmap='jet',s=1)
ax2.axis('square')
fig.colorbar(pcm2,ax=ax2,shrink=0.6)


#%% apprenttissage de T* avec grilles raffinées

# dataset de test

N_test = 100
x_test = np.linspace(-2,2,N_test)
X_test,Y_test = np.meshgrid(x_test,x_test)
X_test,Y_test = X_test.flatten(),Y_test.flatten()
x_true = np.stack((X_test,Y_test),-1)

Z1_test,Z2_test,Z3_test = T_direct(X_test,Y_test)
predict_Z = np.stack((Z1_test,Z2_test,Z3_test),-1)

x = np.linspace(-2,2,N_test)
y = np.zeros(N_test)
z1,z2,z3 = T_direct(x,y)
Traj = np.stack((z1,z2,z3),-1)

# apprentissage de T1* sur grid1

Z1,Z2,Z3 = T_direct(X1,Y1)
data_in = np.stack((Z1,Z2,Z3),-1)
data_out = X1

mlp_reg1 = MLPRegressor(hidden_layer_sizes=(10,),
                       max_iter = 200,activation = 'relu',
                       solver = 'adam')

mlp_reg1.fit(data_in,data_out)

# apprentissage de T2* sur grid2

Z1,Z2,Z3 = T_direct(X2,Y2)
data_in = np.stack((Z1,Z2,Z3),-1)
data_out = Y2

mlp_reg2 = MLPRegressor(hidden_layer_sizes=(10,),
                       max_iter = 200,activation = 'relu',
                       solver = 'adam')

mlp_reg2.fit(data_in,data_out)

# predict

x_pred1 = mlp_reg1.predict(predict_Z)
x_pred2 = mlp_reg2.predict(predict_Z)
traj_pred1 = mlp_reg1.predict(Traj)
traj_pred2 = mlp_reg2.predict(Traj)

erreur = np.sqrt((x_pred1 - x_true[:,0])**2+(x_pred2 - x_true[:,1])**2)

u1 = ((x_true[:,0] - x_pred1)** 2).sum()
v1 = ((x_true[:,0] - x_true[:,0].mean()) ** 2).sum()
score1 = 1-(u1/v1)
u2 = ((x_true[:,1] - x_pred2)** 2).sum()
v2 = ((x_true[:,1] - x_true[:,1].mean()) ** 2).sum()
score2 = 1-(u2/v2)

print('scores raffines :',score1,score2)

plt.figure()
plt.scatter(X_test,Y_test,c=erreur,cmap='jet')
plt.axis('square')
plt.title('distance entre x et x estimé')
plt.colorbar()


#%% données d'apprentissage uniformes équivalentes

N1 = int(np.sqrt(len(grid1)))
N2 = int(np.sqrt(len(grid2)))

g1 = init_grid([2,2,N1,N1,-1,-1])
g2 = init_grid([2,2,N2,N2,-1,-1])

X1,Y1 = coordinate(g1)
X2,Y2 = coordinate(g2)

dZ1 = dT_dz(X1,Y1)

dZ2 = dT_dz(X2,Y2)

dZ1_norme = np.sqrt(dZ1[:,0]**2 + dZ1[:,1]**2 + dZ1[:,2]**2)
dZ2_norme = np.sqrt(dZ2[:,3]**2 + dZ2[:,4]**2 + dZ2[:,5]**2)

_,(ax1,ax2) = plt.subplots(1,2)

ax1.scatter(X1,Y1,c=dZ1_norme,cmap='jet',s=1)
ax1.axis('square')

ax2.scatter(X2,Y2,c=dZ2_norme,cmap='jet',s=1)
ax2.axis('square')


#%% apprentissage de T* avec grilles uniformes

# apprentissage de T1* sur la première grille uniforme g1

Z1,Z2,Z3 = T_direct(X1,Y1)
data_in = np.stack((Z1,Z2,Z3),-1)
data_out = X1

mlp_reg_uni1 = MLPRegressor(hidden_layer_sizes=(10,),
                       max_iter = 200,activation = 'relu',
                       solver = 'adam')

mlp_reg_uni1.fit(data_in,data_out)

# apprentissage de T2* sur la deuxième grille unifome g2

Z1,Z2,Z3 = T_direct(X2,Y2)
data_in = np.stack((Z1,Z2,Z3),-1)
data_out = Y2

mlp_reg_uni2 = MLPRegressor(hidden_layer_sizes=(10,),
                       max_iter = 200,activation = 'relu',
                       solver = 'adam')

mlp_reg_uni2.fit(data_in,data_out)

# prédiction

x_pred1 = mlp_reg_uni1.predict(predict_Z)
x_pred2 = mlp_reg_uni2.predict(predict_Z)

erreur1 = np.abs(x_pred1 - x_true[:,0])
erreur2 = np.abs(x_pred2 - x_true[:,1])

u1 = ((x_true[:,0] - x_pred1)** 2).sum()
v1 = ((x_true[:,0] - x_true[:,0].mean()) ** 2).sum()
score1 = 1-(u1/v1)
u2 = ((x_true[:,1] - x_pred2)** 2).sum()
v2 = ((x_true[:,1] - x_true[:,1].mean()) ** 2).sum()
score2 = 1-(u2/v2)

print('scores uniformes :',score1,score2)
fig,(ax1,ax2) = plt.subplots(1,2)

pcm1 = ax1.scatter(X_test,Y_test,c=erreur1,cmap='jet')
ax1.axis('square')
fig.colorbar(pcm1,ax=ax1,shrink=0.6)

pcm2 = ax2.scatter(X_test,Y_test,c=erreur2,cmap='jet')
ax2.axis('square')
fig.colorbar(pcm2,ax=ax2,shrink=0.62)


#%% résultat exact 

_,(ax1,ax2) = plt.subplots(1,2)
ax1.scatter(X_test,Y_test,c=x_true[:,0],cmap='jet')
ax1.axis('square')

ax2.scatter(X_test,Y_test,c=x_true[:,1],cmap='jet')
ax2.axis('square')


