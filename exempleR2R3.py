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

def eq_uniform_data(X_mesh):
    N_uni = int(np.sqrt(X_mesh.shape[0]))
    x_uni = np.linspace(-1,1,N_uni)
    X_uni,Y_uni = np.meshgrid(x_uni,x_uni)
    X_uni = X_uni.flatten()
    Y_uni = Y_uni.flatten()
    mesh_uni = np.stack((X_uni,Y_uni),-1)
    data_uni = T_direct(X_uni,Y_uni)
    return mesh_uni,data_uni

#%% Gération des données
geometry = [2,2,30,30,-1,-1]

grid = init_grid(geometry)
X,Y = coordinate(grid)
Z = T_direct(X,Y)

titres_T = [r'$T_1(x)$',
            r'$T_2(x)$',
            r'$T_3(x)$']
fig,ax = plt.subplots(1,3,figsize=(17,4))
for i in range(3):
    pc1 = ax[i].scatter(X,Y,c=Z[:,i],cmap = 'jet')
    ax[i].axis('square')
    ax[i].set_xlabel(r'$x1$')
    ax[i].set_ylabel(r'$x2$')
    ax[i].set_title(titres_T[i])
    fig.colorbar(pc1,ax=ax[i])


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

grid1 = iterate_grid(grid1,Z1[:,0],True)
grid2 = iterate_grid(grid2,Z2[:,1],True)
grid3 = iterate_grid(grid3,Z3[:,2],True)

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

_,ax = plt.subplots(1,3)
for i in range(3):
    mesh,data = eq_uniform_data(XX[i])
    data = data[:,i]
    ax[i].scatter(mesh[:,0],mesh[:,1],c = data,s=1,cmap = 'jet')
    ax[i].axis('square')

#%% dataset de test
Nt = 100
x = np.linspace(-1,1,Nt)
Xt2,Xt1 = np.meshgrid(x,x)
Xt = np.stack((Xt1,Xt2),-1)
Xt = Xt.reshape(Nt*Nt,2)
y_exact = T_direct(Xt1.flatten(),Xt2.flatten())

axe = 0

#%% trajectoire de test
theta = np.linspace(0,2*np.pi,Nt)
r = 0.5
x,y = r*np.cos(theta),r*np.sin(theta)
Ct = np.stack((x,y),-1)
Ct = Ct.reshape(Nt,2)

#%% apprentissage de T sur grille non uniforme
data_in = np.stack((XX[axe],YY[axe]),-1)
data_out = ZZ[axe]
# fit
mlp_reg = MLPRegressor(hidden_layer_sizes=(50,20),
                       max_iter = 1000,activation = 'relu',
                       solver = 'adam')

mlp_reg.fit(data_in,data_out)

#%% prediction
y_pred_nu = mlp_reg.predict(Xt)
y_pred_tr_nu = mlp_reg.predict(Ct)
axes = [axe]
u = ((y_exact[:,axes] - y_pred_nu[:,axes])** 2).sum()
v = ((y_exact[:,axes] - y_exact[:,axes].mean()) ** 2).sum()
score = 1-(u/v)
print('score nu :',score)

#%% affichage
# for i in range(3):   
#     plt.figure()
#     plt.scatter(Xt1.flatten(),Xt2.flatten(),c = np.log10(np.abs(y_pred_nu[:,i]-y_exact[:,i])),cmap = 'jet')
#     plt.axis('square')
#     plt.colorbar()

_,ax = plt.subplots(1,3)
for i in range(3):
    ax[i].scatter(Xt1.flatten(),Xt2.flatten(),c = y_pred_nu[:,i],cmap = 'jet')
    ax[i].axis('square')
#%% apprentissage de T sur grille uniforme
mesh_uni,data_uni = eq_uniform_data(XX[axe])

_,ax = plt.subplots(1,3)
for i in range(3):
    ax[i].scatter(mesh_uni[:,0],mesh_uni[:,1],c = data_uni[:,i],s=1,cmap = 'jet')
    ax[i].axis('square')
# fit
mlp_reg_uni = MLPRegressor(hidden_layer_sizes=(50,20),
                       max_iter = 1000,activation = 'relu',
                       solver = 'adam')

mlp_reg_uni.fit(mesh_uni,data_uni)

#%% prediction
y_pred_u = mlp_reg_uni.predict(Xt)
y_pred_tr_u = mlp_reg_uni.predict(Ct)
axes = [2]
u = ((y_exact[:,axes] - y_pred_u[:,axes])** 2).sum()
v = ((y_exact[:,axes] - y_exact[:,axes].mean()) ** 2).sum()
score = 1-(u/v)
print('score u:',score)

#%% affichage
# for i in range(3):   
#     plt.figure()
#     plt.scatter(Xt1.flatten(),Xt2.flatten(),c = np.log10(np.abs(y_pred_u[:,i]-y_exact[:,i])),cmap = 'jet')
#     plt.axis('square')
#     plt.colorbar()

_,ax = plt.subplots(1,3)
for i in range(3):
    ax[i].scatter(Xt1.flatten(),Xt2.flatten(),c = y_pred_u[:,i],cmap = 'jet')
    ax[i].axis('square')
#%% loss curve
plt.plot(mlp_reg.loss_curve_,'r')
plt.plot(mlp_reg_uni.loss_curve_,'b')
plt.legend(['non uniforme','uniforme'])


#%% apprentissage de T*
axe = 0
data_in = ZZ[axe]
data_out = np.stack((XX[axe],YY[axe]),-1)

mlp_inv = MLPRegressor(hidden_layer_sizes=(50,50),
                       max_iter = 1000,activation = 'relu',
                       solver = 'adam')

mlp_inv.fit(data_in,data_out)

#%% prediction
traj_pred_nu = mlp_inv.predict(y_pred_tr_nu)
traj_pred_u = mlp_inv.predict(y_pred_tr_u)
x_pred = mlp_inv.predict(y_pred_u)
erreur = (Xt-x_pred)**2
erreur = (erreur[:,0]+erreur[:,1])/erreur.shape[0]
print(np.sqrt(np.sum(erreur)))
plt.scatter(Xt[:,0],Xt[:,1],c=np.log10(erreur),cmap='jet')
plt.axis('square')
plt.colorbar()


#%% apprentissage en parallèle de T sur chaque dimension
Prediction = []
Score = []
Erreur = []
method = 'raffinement'
for iz in range(len(ZZ)):
    #instantiate the network
    mlp_reg = MLPRegressor(hidden_layer_sizes=(50,10),
                           max_iter = 3000,activation = 'relu',
                           solver = 'adam')
    if method == 'raffinement':
        data_in = np.stack((XX[iz],YY[iz]),-1)
        data_out = ZZ[iz][:,iz]
        
    if method == 'uniforme':
        data_in,data_out = eq_uniform_data(XX[iz])
        # data_in,data_out = eq_uniform_data(XX[axe])
        data_out = data_out[:,iz]
        
    #fit data
    mlp_reg.fit(data_in,data_out)
    #predict
    z_pred = mlp_reg.predict(Xt)
    Prediction.append(z_pred)
    #score
    u = ((y_exact[:,iz] - z_pred)** 2).sum()
    v = ((y_exact[:,iz] - y_exact[:,iz].mean()) ** 2).sum()
    score = 1-(u/v)
    Score.append(score)
    #erreur
    err = np.log10(np.abs(z_pred-y_exact[:,iz]))

    Erreur.append(err)
    
titres_T = [r'$\hat T_1(x)$    Score : %f'%Score[0],
            r'$\hat T_2(x)$    Score : %f'%Score[1],
            r'$\hat T_3(x)$    Score : %f'%Score[2]]

_,ax = plt.subplots(1,3,figsize=(17,4))
for i in range(3):
    pc1 = ax[i].scatter(Xt1.flatten(),Xt2.flatten(),c = Prediction[i],cmap = 'jet')
    ax[i].axis('square')
    ax[i].set_xlabel(r'$x1$')
    ax[i].set_ylabel(r'$x2$')
    ax[i].set_title(titres_T[i])
    fig.colorbar(pc1,ax=ax[i])

print(method,' :',Score)