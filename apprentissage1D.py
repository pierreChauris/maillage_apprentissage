# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:56:00 2022

@author: pchauris

Objectif : quantifier l'intérêt du raffinement de maillage sur l'apprentissage par réseau de neurone
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from raffinement import *

#%% définition des données

def f(X,Y):
    return np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2)


geometry = [10,10,50,50,0,0]
grid = init_grid(geometry)
X,Y = coordinate(grid)

# plt.scatter(X,Y,c=f(X,Y),cmap='jet',s=1)
# # for cell in grid:
# #     plot_cell(cell)
# plt.axis('square')
# plt.colorbar()
# plt.title('transformation à apprendre')

#%% iterations de raffinement

niter = 2
for i in range(niter):
    grid = iterate_grid(grid, f(X,Y), False)
    X,Y = coordinate(grid)
    
# plt.scatter(X,Y,c=f(X,Y),cmap='jet',s=1)
# # for cell in grid:
# #     plot_cell(cell)
# plt.axis('square')
# plt.colorbar()

print('taille de la grille :',len(grid))
#%% données d'entrainement non uniforme

train_X_nu = np.stack((X,Y),-1)
train_Z_nu = f(X,Y)

sc=StandardScaler()
scaler_nu = sc.fit(train_X_nu)
train_X_nu = scaler_nu.transform(train_X_nu)

#%% grille uniforme équivalente

N_uni = int(np.sqrt(len(grid)))
x_uni = np.linspace(0,10,N_uni)
X_uni,Y_uni = np.meshgrid(x_uni,x_uni)
X_uni,Y_uni = X_uni.flatten(),Y_uni.flatten()
train_X_uni = np.stack((X_uni,Y_uni),-1)
train_Z_uni = f(X_uni,Y_uni)

sc=StandardScaler()
scaler_uni = sc.fit(train_X_uni)
train_X_uni = scaler_uni.transform(train_X_uni)

#%% apprentissage uniforme et non uniforme

data_in = train_X_uni
data_out = train_Z_uni
# fit
mlp_reg_uni = MLPRegressor(hidden_layer_sizes=(30,10),
                       max_iter = 1000,activation = 'relu',
                       solver = 'adam')

mlp_reg_uni.fit(data_in,data_out)

data_in = train_X_nu
data_out = train_Z_nu
# fit
mlp_reg_nu = MLPRegressor(hidden_layer_sizes=(30,10),
                       max_iter = 1000,activation = 'relu',
                       solver = 'adam')

mlp_reg_nu.fit(data_in,data_out)

#%% dataset de test

N_test = 200
x_test = np.linspace(0,10,N_test)
X_test,Y_test = np.meshgrid(x_test,x_test)
X_test,Y_test = X_test.flatten(),Y_test.flatten()
predict_X = np.stack((X_test,Y_test),-1)
predict_X_uni = scaler_uni.transform(predict_X)
predict_X_nu = scaler_nu.transform(predict_X)

Z_exact = f(X_test,Y_test)
Z_predict_uni = np.abs(mlp_reg_uni.predict(predict_X_uni))
Z_predict_nu = np.abs(mlp_reg_nu.predict(predict_X_nu))

u = ((Z_exact - Z_predict_uni)** 2).sum()
v = ((Z_exact - Z_exact.mean()) ** 2).sum()
score_uniforme = 1-(u/v)

u = ((Z_exact - Z_predict_nu)** 2).sum()
v = ((Z_exact - Z_exact.mean()) ** 2).sum()
score_raffine = 1-(u/v)

#%% résultats

plt.figure()
plt.scatter(X_test,Y_test,c=Z_predict_uni,s=1,cmap='jet')
plt.axis('square')
plt.colorbar()
plt.title('prédiction à partir du dataset uniforme')
plt.figure()
plt.scatter(X_test,Y_test,c=Z_predict_nu,s=1,cmap='jet')
plt.axis('square')
plt.colorbar()
plt.title('prédiction à partir du dataset raffiné')

print('score uniforme :',score_uniforme)
print('score raffine :',score_raffine)
