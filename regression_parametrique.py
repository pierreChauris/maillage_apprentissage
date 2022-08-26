# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:18:32 2022

@author: pchauris
"""

"Objectif : Intérêt du raffinement pour la régression paramétrique d'un système linéaire des paramètres"
"Impact du bruit sur la sortie"

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
    

def crit_sequence(grid,theta):
    return gradient(grid,theta)

def alpha_sequence(grid,theta):
    sequence = crit_sequence(grid,theta)
    return np.linspace(0,sequence.max(),len(grid))


def distrib_sequence(grid,theta):
    alpha = alpha_sequence(grid,theta)
    crit = crit_sequence(grid,theta)
    distribution = []
    for j in range(alpha.size):
        dj = np.count_nonzero(crit>alpha[j])
        distribution.append(dj)
    return np.array(distribution)


def auto_threshold(grid,theta):
    alpha = alpha_sequence(grid,theta)
    distribution = distrib_sequence(grid,theta)
    f = alpha*distribution
    #maximum global
    fmax = np.max(f)
    idmax = np.where(f == fmax)
    idmax = idmax[0][0]
    #alpha 
    alphamax = alpha[idmax]
    return alphamax

def iterate_grid(grid,theta):
    alpha = auto_threshold(grid,theta)
    crit = crit_sequence(grid,theta)
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


def R_matrice(X,Y):
    return np.stack((X**3,Y**3,X**2,Y**2,X*Y,X,Y,np.ones(X.size)),-1)

def P_theta(theta,X,Y):
    R = R_matrice(X,Y)
    return np.dot(R,theta)

def uniform_grid(dimensions,npts):
    Lx,Ly,Ox,Oy = dimensions
    N = int(np.sqrt(npts)) + 1
    x,y = np.linspace(Ox,Ox+Lx,N), np.linspace(Oy,Oy+Ly,N)
    X,Y = np.meshgrid(x,y)
    while X.size != npts:
        ind = np.random.randint(0,X.size)
        X = np.delete(X,ind)
        Y = np.delete(Y,ind)
    return X.flatten(),Y.flatten()

def gradient(grid,theta):
    a1,a2,a3,a4,a5,a6,a7,a8 = theta
    X,Y = coordinate(grid)
    dPdx = 3*a1*X**2 + 2*a3*X + a5*Y + a6*np.ones(X.size)
    dPdy = 3*a2*Y**2 + 2*a4*Y + a5*X + a7*np.ones(X.size)
    return np.sqrt(dPdx**2 + dPdy**2)

#%%

geometry = [2,2,30,30,-1,-1]
grid = init_grid(geometry)
X,Y = coordinate(grid)
theta = np.array([1,-1,1,0,1,1,-1,1])
Z = P_theta(theta,X,Y)

plt.figure()
plt.scatter(X,Y,c=Z,cmap='jet')
plt.colorbar()
plt.axis('square')

# résolution

R = R_matrice(X,Y)
theta_barre = np.dot(np.linalg.pinv(R),Z)
Z_barre = np.dot(R,theta_barre)

print(theta_barre)

# ajout du bruit

bruit = np.random.normal(0, 0.1, len(grid))

Z_bruite = Z + bruit

# plt.figure()
# plt.scatter(X,Y,c=Z_bruite,cmap='jet')
# plt.colorbar()
# plt.axis('square')

theta_star = np.dot(np.linalg.pinv(R),Z_bruite)
Z_star = np.dot(R,theta_star) 

# plt.figure()
# plt.scatter(X,Y,c=Z_star,cmap='jet')
# plt.colorbar()
# plt.axis('square')

print(theta_star)

critere = crit_sequence(grid,theta)

plt.figure()
plt.scatter(X,Y,c=critere,cmap='jet')
plt.colorbar()
plt.axis('square')

#%% comparaison uniforme - raffinement
geometry = [2,2,20,20,-1,-1]
grid = init_grid(geometry)
X,Y = coordinate(grid)
theta = np.array([1,1,1,0,1,1,-1,1])
critere = crit_sequence(grid,theta)

plt.figure()
plt.scatter(X,Y,c=critere,cmap='jet')
plt.colorbar()
plt.axis('square')

grid2 = iterate_grid(grid,theta)
bruit = np.random.normal(0, 3, len(grid2))

# generation des données
Xnu,Ynu = coordinate(grid2)
Znu = P_theta(theta,Xnu,Ynu) + bruit


Xu,Yu = uniform_grid([2,2,-1,-1],len(grid2))
Zu = P_theta(theta,Xu,Yu) + bruit

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
pc1 = ax1.scatter(Xu,Yu,c=Zu,cmap='jet',s=1)
ax1.axis('square')
fig.colorbar(pc1,ax=ax1)
pc2 = ax2.scatter(Xnu,Ynu,c=Znu,cmap='jet',s=1)
ax2.axis('square')
fig.colorbar(pc2,ax=ax2)

# estimation
Ru = R_matrice(Xu,Yu)
theta_u = np.dot(np.linalg.pinv(Ru),Zu)

Rnu = R_matrice(Xnu,Ynu)
theta_nu = np.dot(np.linalg.pinv(Rnu),Znu)

err_u = np.linalg.norm(theta - theta_u)
err_nu = np.linalg.norm(theta - theta_nu)

print('theta exact :',theta)
print('theta uniforme :',theta_u)
print('theta raffiné :',theta_nu)

print('erreur sur theta avec les données uniformes :',err_u)
print('erreur sur theta avec les données raffinées :',err_nu)


x = np.arange(8)
data = np.stack((np.abs(theta-theta_u),np.abs(theta-theta_nu)))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x + 0.00, data[0], color = 'r', width = 0.25)
ax.bar(x + 0.25, data[1], color = 'g', width = 0.25)
ax.legend(labels=['Erreur uniforme', 'Erreur raffinement'])

#%% batch sur le bruit

Err_u,Err_nu = [],[]
for _ in range(1000):
    # generation des données
    bruit = np.random.normal(0, 3, len(grid2))

    Znu = P_theta(theta,Xnu,Ynu) + bruit

    Zu = P_theta(theta,Xu,Yu) + bruit
    
    # estimation
    Ru = R_matrice(Xu,Yu)
    theta_u = np.dot(np.linalg.pinv(Ru),Zu)

    Rnu = R_matrice(Xnu,Ynu)
    theta_nu = np.dot(np.linalg.pinv(Rnu),Znu)

    Err_u.append(np.abs(theta-theta_u))
    Err_nu.append(np.abs(theta-theta_nu))

Err_u,Err_nu = np.array(Err_u),np.array(Err_nu)
data = np.stack((np.mean(Err_u,axis=0),np.mean(Err_nu,axis=0)))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x + 0.00, data[0], color = 'r', width = 0.25)
ax.bar(x + 0.25, data[1], color = 'g', width = 0.25)
ax.legend(labels=['Erreur uniforme', 'Erreur raffinement'])

#%% batch sur la variance du bruit

Err_u,Err_nu = [],[]
Var = np.linspace(0,2,100)
for var in Var:
    # generation des données
    bruit = np.random.normal(0, var, len(grid2))

    Znu = P_theta(theta,Xnu,Ynu) + bruit

    Zu = P_theta(theta,Xu,Yu) + bruit
    
    # estimation
    Ru = R_matrice(Xu,Yu)
    theta_u = np.dot(np.linalg.pinv(Ru),Zu)

    Rnu = R_matrice(Xnu,Ynu)
    theta_nu = np.dot(np.linalg.pinv(Rnu),Znu)

    Err_u.append(np.linalg.norm(theta-theta_u))
    Err_nu.append(np.linalg.norm(theta-theta_nu))

plt.plot(Var,Err_u)
plt.plot(Var,Err_nu)
plt.legend(['Erreur uniforme', 'Erreur raffinement'])

#%% batch sur la taille de la grille

Err_u,Err_nu = [],[]

for N in range(5,40):
    geometry = [2,2,N,N,-1,-1]
    grid = init_grid(geometry)
    theta = np.array([1,1,1,0,1,1,-1,1])

    grid2 = iterate_grid(grid,theta)
    bruit = np.random.normal(0, 0.1, len(grid2))

    # generation des données
    Xnu,Ynu = coordinate(grid2)
    Znu = P_theta(theta,Xnu,Ynu) + bruit


    Xu,Yu = uniform_grid([2,2,-1,-1],len(grid2))
    Zu = P_theta(theta,Xu,Yu) + bruit

    # estimation
    Ru = R_matrice(Xu,Yu)
    theta_u = np.dot(np.linalg.pinv(Ru),Zu)

    Rnu = R_matrice(Xnu,Ynu)
    theta_nu = np.dot(np.linalg.pinv(Rnu),Znu)

    err_u = np.linalg.norm(theta - theta_u)
    err_nu = np.linalg.norm(theta - theta_nu)
    Err_u.append(err_u)
    Err_nu.append(err_nu)


plt.plot(np.arange(5,40),Err_u)
plt.plot(np.arange(5,40),Err_nu)
plt.legend(['Erreur uniforme', 'Erreur raffinement'])
