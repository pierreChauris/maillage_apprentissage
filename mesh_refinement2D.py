# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:57:24 2022

@author: pchauris
"""
#%% imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from findiff import FinDiff
import time
#%% def classe et fonctions

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

def f(X,Y):
    #return np.exp(-0.4*(X-5)**2 - 0.4*(Y-5)**2)
    return np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2)

def emp_grad(cell):
    x1,y1 = cell.center()
    level = cell.level
    dx = L/(N*2**(level-1))
    x0,y0 = x1 - dx/2 , y1 - dx/2
    Dfx = f(x0+dx,y0) - f(x0,y0)
    Dfy = f(x0,y0+dx) - f(x0,y0)
    return np.abs(Dfx/dx),np.abs(Dfy/dx)


def crit_sequence(grid):
    res = []
    for cell in grid:
        gx,gy = emp_grad(cell)
        res.append(np.sqrt(gx**2+gy**2))
        # res.append(max(gx,0))
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
        gx,gy = emp_grad(cell)
        if np.sqrt(gx**2+gy**2) > alpha:
        # if max(gx,0) > alpha:
            C00,C01,C10,C11 = cell.split()
            new_grid.remove(cell)
            new_grid.insert(k,C11)
            new_grid.insert(k,C10)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
    return new_grid

#%% raffinement
N = 11
L = 10
x10,x20 = 0,0

grid = init_grid(N)
niter = 3
X,Y = cen(grid)
plt.figure()
plt.scatter(X,Y,c = f(X,Y),s = 1,cmap = 'jet')
plt.axis('square')
plt.colorbar()

for _ in range(niter):
    alpha = auto_threshold(grid)
    grid = iterate_grid(grid,alpha)
    X,Y = cen(grid)
    plt.figure()
    plt.scatter(X,Y,c = f(X,Y),s = 1,cmap = 'jet')
    plt.axis('square')
    plt.colorbar()

#%% génération grille uniforme

N_uni = int(np.sqrt(len(grid)))
x_uni = np.linspace(0,L,N_uni)
X_uni,Y_uni = np.meshgrid(x_uni,x_uni)
X_uni = X_uni.reshape(N_uni*N_uni)
Y_uni = Y_uni.reshape(N_uni*N_uni)
grid_uni = np.stack((X_uni,Y_uni),-1)
data_uni = f(X_uni,Y_uni)

#%% generation grille non uniforme

X_nu,Y_nu = cen(grid)
grid_nu = np.stack((X_nu,Y_nu),-1)
data_nu = f(X_nu,Y_nu)

#%% grille de test

Npred = 80
x_pred = np.linspace(0,L,Npred)
X_pred,Y_pred = np.meshgrid(x_pred,x_pred)
X_pred = X_pred.reshape(Npred*Npred)
Y_pred = Y_pred.reshape(Npred*Npred)
grid_pred = np.stack((X_pred,Y_pred),-1)
#grid_pred = scaler.transform(grid_pred)
z_exact = f(X_pred,Y_pred)

#%% entrainement et estimation sur grille uniforme

# scale and fit
mlp_reg = MLPRegressor(hidden_layer_sizes=(200,200,100),
                       max_iter = 500,activation = 'relu',
                       solver = 'adam')
sc=StandardScaler()
scaler = sc.fit(grid_uni)
#grid_uni = scaler.transform(grid_uni)

mlp_reg.fit(grid_uni,data_uni)

z_pred = mlp_reg.predict(grid_pred)
err_pred = np.abs(z_exact-z_pred)
print('erreur uniforme :',np.linalg.norm(err_pred))

#%% entrainement et estimation sur grille non uniforme

# scale and fit
mlp_reg_nu = MLPRegressor(hidden_layer_sizes=(200,200,100),
                       max_iter = 500,activation = 'relu',
                       solver = 'adam')
sc=StandardScaler()
scaler = sc.fit(grid_nu)
#grid_nu = scaler.transform(grid_nu)

mlp_reg_nu.fit(grid_nu,data_nu)

# predict over uniform grid
z_pred_nu = mlp_reg_nu.predict(grid_pred)
err_pred_nu = np.abs(z_exact-z_pred_nu)
print('erreur non uniforme :',np.linalg.norm(err_pred_nu))
#%% affichage

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6.9))
colmap = 'jet'
im = axes[0,0].scatter(X_uni,Y_uni,c = f(X_uni,Y_uni),s = 1, cmap=colmap)
cb_ax = fig.add_axes([0.24, 0.527, 0.01, 0.35])
cbar = fig.colorbar(im, cax=cb_ax)
axes[0,0].set_title('dataset uniforme')

im = axes[0,1].scatter(X_pred,Y_pred,c = z_pred,s = 1, cmap=colmap)
cb_ax = fig.add_axes([0.524, 0.527, 0.01, 0.35])
cbar = fig.colorbar(im, cax=cb_ax)
axes[0,1].set_title('prédiction ')

im = axes[0,2].scatter(X_pred,Y_pred,c = err_pred,s = 1, cmap=colmap)
cb_ax = fig.add_axes([0.795, 0.527, 0.01, 0.35])
cbar = fig.colorbar(im, cax=cb_ax)
axes[0,2].set_title('erreur absolue')

im = axes[1,0].scatter(X_nu,Y_nu,c = f(X_nu,Y_nu),s = 1, cmap=colmap)
cb_ax = fig.add_axes([0.24, 0.1242, 0.01, 0.35])
cbar = fig.colorbar(im, cax=cb_ax)
axes[1,0].set_title('dataset non uniforme')

im = axes[1,1].scatter(X_pred,Y_pred,c = z_pred_nu,s = 1, cmap=colmap)
cb_ax = fig.add_axes([0.524, 0.1242, 0.01, 0.35])
cbar = fig.colorbar(im, cax=cb_ax)
axes[1,1].set_title('prédiction')

im = axes[1,2].scatter(X_pred,Y_pred,c = err_pred_nu,s = 1, cmap=colmap)
cb_ax = fig.add_axes([0.795, 0.1242, 0.01, 0.35])
cbar = fig.colorbar(im, cax=cb_ax)
axes[1,2].set_title('erreur absolue')


for ax in axes.flat:
    ax.set_axis_off()
    ax.axis('square')


fig.subplots_adjust(bottom=0.1, top=0.9, left=0., right=0.8,
                    wspace=0.2, hspace=0.02)

plt.show()

#%% affichage 2

fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 6.9))
im = ax1.scatter(X_pred,Y_pred,c = z_pred,s = 1, cmap=colmap)
cb_ax = fig.add_axes([0.6, 0.66, 0.01, 0.22])
cbar = fig.colorbar(im, cax=cb_ax)
ax1.set_title('prédiction uniforme')
ax1.axis('square')

im = ax2.scatter(X_pred,Y_pred,c = z_pred_nu,s = 1, cmap=colmap)
cb_ax = fig.add_axes([0.6, 0.39, 0.01, 0.22])
cbar = fig.colorbar(im, cax=cb_ax)
ax2.set_title('prédiction non uniforme')
ax2.axis('square')


im = ax3.scatter(X_pred,Y_pred,c = f(X_pred,Y_pred), cmap=colmap)
cb_ax = fig.add_axes([0.6, 0.126, 0.01, 0.22])
cbar = fig.colorbar(im, cax=cb_ax)
ax3.set_title('solution exacte')
ax3.axis('square')
#%% gradient empirique
grad = []
start = time.time()
for cell in grid:
    gx,gy = emp_grad(cell)
    grad.append(np.sqrt(gx**2 + gy**2))
end = time.time()
time_emp_grad = end-start
print('temps calcul à la main :',time_emp_grad)
plt.figure()
plt.scatter(X_nu,Y_nu,c = grad,s = 1,cmap = 'jet')
plt.axis('square')
plt.colorbar()

#%% gradient avec Findiff
Z = f(X_uni,Y_uni).reshape(N_uni,N_uni)
start = time.time()
d_dx1 = FinDiff(0,1)
d_dx2 = FinDiff(1,1)
dZ_dx1 = d_dx1(Z)
dZ_dx2 = d_dx2(Z)
gr = np.sqrt(dZ_dx1**2+dZ_dx2**2)
end = time.time()
time_findiff = end-start
print('temps calcul findiff :',time_findiff)
plt.figure()
plt.scatter(X_uni,Y_uni,c = gr,s = 1,cmap = 'jet')
plt.axis('square')
plt.colorbar()

#%% erreur en fonction de la taille de la grille

# N_liste = np.arange(8,45)
# res_uni,res_nu,grid_len = [],[],[]
# for N in N_liste:
#     print(N)
#     grid = init_grid(N)
#     for _ in range(2):
#         alpha = auto_threshold(grid)
#         grid = iterate_grid(grid,alpha)
#     grid_len.append(len(grid))
#     # dataset non uniforme
#     X_nu,Y_nu = cen(grid)
#     grid_nu = np.stack((X_nu,Y_nu),-1)
#     data_nu = f(X_nu,Y_nu)
#     # dataset uniforme
#     N_uni = int(np.sqrt(len(grid)))
#     x_uni = np.linspace(0,L,N_uni)
#     X_uni,Y_uni = np.meshgrid(x_uni,x_uni)
#     X_uni = X_uni.reshape(N_uni*N_uni)
#     Y_uni = Y_uni.reshape(N_uni*N_uni)
#     grid_uni = np.stack((X_uni,Y_uni),-1)
#     data_uni = f(X_uni,Y_uni)
#     # grille de test
#     Npred = 80
#     x_pred = np.linspace(0,L,Npred)
#     X_pred,Y_pred = np.meshgrid(x_pred,x_pred)
#     X_pred = X_pred.reshape(Npred*Npred)
#     Y_pred = Y_pred.reshape(Npred*Npred)
#     grid_pred = np.stack((X_pred,Y_pred),-1)
#     z_exact = f(X_pred,Y_pred)

#     # apprentissage uniforme
#     mlp_reg = MLPRegressor(hidden_layer_sizes=(200,200,100),
#                            max_iter = 500,activation = 'relu',
#                            solver = 'adam')
#     mlp_reg.fit(grid_uni,data_uni)
#     z_pred = mlp_reg.predict(grid_pred)
#     err_pred = np.abs(z_exact-z_pred)
#     res_uni.append(np.linalg.norm(err_pred))
    
#     # apprentissage non uniforme
#     mlp_reg_nu = MLPRegressor(hidden_layer_sizes=(200,200,100),
#                            max_iter = 500,activation = 'relu',
#                            solver = 'adam')
#     mlp_reg_nu.fit(grid_nu,data_nu)
#     z_pred_nu = mlp_reg_nu.predict(grid_pred)
#     err_pred_nu = np.abs(z_exact-z_pred_nu)
#     res_nu.append(np.linalg.norm(err_pred_nu))
    
#%% affichage
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

plt.plot(grid_len,smooth(res_uni,5),'-o')
plt.plot(grid_len,smooth(res_nu,5),'-o')
plt.axvline(x = 1250,linestyle = '--',color = 'r')
plt.axvline(x = 3050,linestyle = '--',color = 'r')
plt.title('norme de l erreur d estimation en fonction du nombre de données')
plt.legend(['grille uniforme','grille non uniforme'])