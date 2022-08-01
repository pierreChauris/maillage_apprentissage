# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:51:05 2022

@author: pchauris
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from raffinement import *
#%%
def T_direct(X,Y):
    s = X**2 + Y**2 + 0.001
    return np.array([s,X/s,Y/s]).T

def T_inverse(Z1,Z2,Z3):
    return np.array([Z1*Z2,Z1*Z3]).T

#%% Gération des données
geometry = [2,2,20,20,-1,-1]

grid = init_grid(geometry)
X,Y = coordinate(grid)
Z = T_direct(X,Y)

_,ax = plt.subplots(1,3)
for i in range(Z.shape[1]):
    ax[i].scatter(X,Y,c = Z[:,i],s=5,cmap = 'jet')
    ax[i].axis('square')


#%% Raffinement des données
grid1 = iterate_grid(grid,Z[:,0],True)
grid2 = iterate_grid(grid,Z[:,1],True)
grid3 = iterate_grid(grid,Z[:,2],True)

grids = [grid1,grid2,grid3]

Xr1,Yr1 = coordinate(grid1)
Xr2,Yr2 = coordinate(grid2)
Xr3,Yr3 = coordinate(grid3)

XX = [Xr1,Xr2,Xr3]
YY = [Yr1,Yr2,Yr3]

Z1 = T_direct(Xr1,Yr1)
Z2 = T_direct(Xr2,Yr2)
Z3 = T_direct(Xr3,Yr3)

ZZ = [Z1,Z2,Z3]

_,(ax1,ax2,ax3) = plt.subplots(1,3)
ax1.scatter(Xr1,Yr1,c=Z1[:,0],s=1,cmap='jet')
ax1.axis('square')
ax2.scatter(Xr2,Yr2,c=Z2[:,1],s=1,cmap='jet')
ax2.axis('square')
ax3.scatter(Xr3,Yr3,c=Z3[:,2],s=1,cmap='jet')
ax3.axis('square')


#%% dataset de test
Nt = 100
x = np.linspace(-1,1,Nt)
Xt2,Xt1 = np.meshgrid(x,x)
Xt = np.stack((Xt1,Xt2),-1)
Xt = Xt.reshape(Nt*Nt,2)
y_exact = T_direct(Xt1.flatten(),Xt2.flatten())

axe = 1

#%%

theta = np.linspace(0,2*np.pi,50)
x,y = 0.5*np.cos(theta),0.5*np.sin(theta)
Ct = np.stack((x,y),-1)
Ct = Ct.reshape(50,2)

plt.scatter(Xt1.flatten(),Xt2.flatten(),c='white')
plt.scatter(x,y,c=T_direct(x,y)[:,0],cmap='jet')
plt.colorbar()
plt.axis('square')
#%% apprentissage de T sur grille non uniforme
data_in = np.stack((XX[axe],YY[axe]),-1)
data_out = ZZ[axe]
# fit
mlp_reg = MLPRegressor(hidden_layer_sizes=(50,10),
                       max_iter = 1000,activation = 'relu',
                       solver = 'adam')

mlp_reg.fit(data_in,data_out)

# prediction
y_pred_nu = mlp_reg.predict(Xt)
y_pred_tr = mlp_reg.predict(Ct)
axes = [axe]
u = ((y_exact[:,axes] - y_pred_nu[:,axes])** 2).sum()
v = ((y_exact[:,axes] - y_exact[:,axes].mean()) ** 2).sum()
score = 1-(u/v)
print('score nu :',score)

#%% affichage
for i in range(3):   
    plt.figure()
    plt.scatter(Xt1.flatten(),Xt2.flatten(),c = np.log10(np.abs(y_pred_nu[:,i]-y_exact[:,i])),cmap = 'jet')
    plt.axis('square')
    plt.colorbar()

_,ax = plt.subplots(1,3)
for i in range(3):
    ax[i].scatter(Xt1.flatten(),Xt2.flatten(),c = y_pred_nu[:,i],cmap = 'jet')
    ax[i].axis('square')
#%% apprentissage de T sur grille uniforme
N_uni = int(np.sqrt(len(grids[axe])))
x_uni = np.linspace(-1,1,N_uni)
X_uni,Y_uni = np.meshgrid(x_uni,x_uni)
X_uni = X_uni.flatten()
Y_uni = Y_uni.flatten()
mesh_uni = np.stack((X_uni,Y_uni),-1)
data_uni = T_direct(X_uni,Y_uni)

# fit
mlp_reg_uni = MLPRegressor(hidden_layer_sizes=(50,10),
                       max_iter = 1000,activation = 'relu',
                       solver = 'adam')

mlp_reg_uni.fit(mesh_uni,data_uni)

# prediction
y_pred_u = mlp_reg_uni.predict(Xt)
axes = [axe]
u = ((y_exact[:,axes] - y_pred_u[:,axes])** 2).sum()
v = ((y_exact[:,axes] - y_exact[:,axes].mean()) ** 2).sum()
score = 1-(u/v)
print('score u:',score)

#%% affichage
for i in range(3):   
    plt.figure()
    plt.scatter(Xt1.flatten(),Xt2.flatten(),c = np.log10(np.abs(y_pred_u[:,i]-y_exact[:,i])),cmap = 'jet')
    plt.axis('square')
    plt.colorbar()

_,ax = plt.subplots(1,3)
for i in range(3):
    ax[i].scatter(Xt1.flatten(),Xt2.flatten(),c = y_pred_u[:,i],cmap = 'jet')
    ax[i].axis('square')
#%% loss curve
plt.plot(mlp_reg.loss_curve_,'r')
plt.plot(mlp_reg_uni.loss_curve_,'b')
plt.legend(['non uniforme','uniforme'])


#%% apprentissage de T*

data_in = ZZ[axe]
data_out = np.stack((XX[axe],YY[axe]),-1)

mlp_inv = MLPRegressor(hidden_layer_sizes=(50,50),
                       max_iter = 1000,activation = 'relu',
                       solver = 'adam')

mlp_inv.fit(data_in,data_out)

#%% prediction
traj_pred = mlp_inv.predict(y_pred_tr)
x_pred = mlp_inv.predict(y_pred_nu)
erreur = (Xt-x_pred)**2
erreur = (erreur[:,0]+erreur[:,1])/erreur.shape[0]
print(np.sqrt(np.sum(erreur)))
plt.scatter(Xt[:,0],Xt[:,1],c=erreur,cmap='jet')
plt.axis('square')
plt.colorbar()

plt.plot(x,y,c='green',linestyle='--')
plt.plot(traj_pred[:,0],traj_pred[:,1],c='r',linestyle='--')
plt.axis('square')
