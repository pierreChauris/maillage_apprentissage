# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:17:19 2022

@author: pchauris
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from raffinement import *


def T_direct(X,Y):
    s = X**2 + Y**2 + 0.01
    return np.array([s,X/s,Y/s]).T

#%% generate artificial data X,Z for training - uniform method

N_train = 23
x_train = np.linspace(-1,1,N_train)
X_train,Y_train = np.meshgrid(x_train,x_train)
X_train,Y_train = X_train.flatten(), Y_train.flatten()
train_data_X_uni = np.stack((X_train,Y_train),-1)
train_data_Z_uni = T_direct(X_train,Y_train)

_,ax = plt.subplots(1,3)
for i in range(3):
    ax[i].scatter(train_data_X_uni[:,0],train_data_X_uni[:,1],c = train_data_Z_uni[:,i],s=1,cmap = 'jet')
    ax[i].axis('square')

#%% generate artificial data X,Z for training - non uniform method

N_init = 20
geometry = [2,2,N_init,N_init,-1,-1]
grid = init_grid(geometry)
X_grid,Y_grid = coordinate(grid)
Z_grid = T_direct(X_grid,Y_grid)

axe = 0
grid = iterate_grid(grid, Z_grid[:,axe], True)
X_grid,Y_grid = coordinate(grid)

train_data_X_nu = np.stack((X_grid,Y_grid),-1)
train_data_Z_nu = T_direct(X_grid,Y_grid)

_,ax = plt.subplots(1,3)
for i in range(3):
    ax[i].scatter(train_data_X_nu[:,0],train_data_X_nu[:,1],c = train_data_Z_nu[:,i],s=1,cmap = 'jet')
    ax[i].axis('square')
    
#%% train T*

data_in = train_data_Z_nu

data_out = train_data_X_nu

mlp_inv = MLPRegressor(
                       max_iter = 300,activation = 'relu',
                       solver = 'adam')

mlp_inv.fit(data_in,data_out)

#%% predict over a test grid

N_test = 100
x_test = np.linspace(-1,1,N_test)
X_test,Y_test = np.meshgrid(x_test,x_test)
X_test,Y_test = X_test.flatten(), Y_test.flatten()
test_data_X = np.stack((X_test,Y_test),-1)
test_data_Z = T_direct(X_test,Y_test)

predict_data_X = mlp_inv.predict(test_data_Z)

erreur = test_data_X - predict_data_X
RMSE = np.sqrt(erreur[:,0]**2 + erreur[:,1]**2)
print('RMSE global :',np.linalg.norm(RMSE))

#%% predict trajectories

theta = np.linspace(0,2*np.pi,N_test)
r = 0.5
x,y = r*np.cos(theta),r*np.sin(theta)
test_traj_X = np.stack((x,y),-1)
test_traj_Z = T_direct(test_traj_X[:,0],test_traj_X[:,1])

predict_traj_X = mlp_inv.predict(test_traj_Z)

#%% affichage

plt.figure()
plt.scatter(X_test,Y_test,c=RMSE,s=1,cmap='jet')
plt.colorbar()
plt.scatter(test_traj_X[:,0],test_traj_X[:,1],c='white',s=5)
plt.scatter(predict_traj_X[:,0],predict_traj_X[:,1],s=5,c='r')
plt.axis('square')
plt.title('RMSE and trajectory prediction')
