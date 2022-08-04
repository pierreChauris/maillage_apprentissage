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

#%% régression paramétrique

def f_param(theta,X,Y):
    x1,y1,s1,x2,y2,s2 = theta
    return np.exp(-1/s1*(X-x1)**2 - 1/s1*(Y-y1)**2) + np.exp(-1/s2*(X-x2)**2 - 1/s2*(Y-y2)**2)

def dJ_dx1(theta,X,Y,Z):
    x1,y1,s1,x2,y2,s2 = theta
    dg_dx1 = -2/s1*(X-x1)*np.exp(-1/s1*(X-x1)**2 - 1/s1*(Y-y1)**2) 
    arg = dg_dx1*(Z-f_param(theta,X,Y))
    return np.sum(arg)/np.size(arg)


def dJ_dy1(theta,X,Y,Z):
    x1,y1,s1,x2,y2,s2 = theta
    dg_dy1 = -2/s1*(Y-y1)*np.exp(-1/s1*(X-x1)**2 - 1/s1*(Y-y1)**2) 
    arg = dg_dy1*(Z-f_param(theta,X,Y))
    return np.sum(arg)/np.size(arg)


def dJ_ds1(theta,X,Y,Z):
    x1,y1,s1,x2,y2,s2 = theta
    dg_ds1 = -((X-x1)**2 + (Y-y1)**2)/(s1**2)*np.exp(-1/s1*(X-x1)**2 - 1/s1*(Y-y1)**2) 
    arg = dg_ds1*(Z-f_param(theta,X,Y))
    return np.sum(arg)/np.size(arg)


def dJ_dx2(theta,X,Y,Z):
    x1,y1,s1,x2,y2,s2 = theta
    dg_dx2 = -2/s2*(X-x2)*np.exp(-1/s2*(X-x2)**2 - 1/s2*(Y-y2)**2) 
    arg = dg_dx2*(Z-f_param(theta,X,Y))
    return np.sum(arg)/np.size(arg)


def dJ_dy2(theta,X,Y,Z):
    x1,y1,s1,x2,y2,s2 = theta
    dg_dy2 = -2/s2*(Y-y2)*np.exp(-1/s2*(X-x2)**2 - 1/s2*(Y-y2)**2) 
    arg = dg_dy2*(Z-f_param(theta,X,Y))
    return np.sum(arg)/np.size(arg)


def dJ_ds2(theta,X,Y,Z):
    x1,y1,s1,x2,y2,s2 = theta
    dg_ds2 = -((X-x2)**2 + (Y-y2)**2)/(s2**2)*np.exp(-1/s2*(X-x2)**2 - 1/s2*(Y-y2)**2)
    arg = dg_ds2*(Z-f_param(theta,X,Y))
    return np.sum(arg)/np.size(arg)


def J_cost(theta,X,Y,Z):
    arg = (Z-f_param(theta,X,Y))**2
    return np.sum(arg)/(2*np.size(arg))

def estimation(X,Y,Z,theta0,alpha,Nmax):
    niter = 0
    res = [theta0]
    dJ = np.array([dJ_dx1(theta0,X,Y,Z),dJ_dy1(theta0,X,Y,Z),dJ_ds1(theta0,X,Y,Z),dJ_dx2(theta0,X,Y,Z),dJ_dy2(theta0,X,Y,Z),dJ_ds2(theta0,X,Y,Z)])
    cost = [J_cost(theta0,X,Y,Z)]
    
    while J_cost(theta0,X,Y,Z) > 10**-4 and niter < Nmax:
        dJ = np.array([dJ_dx1(theta0,X,Y,Z),dJ_dy1(theta0,X,Y,Z),dJ_ds1(theta0,X,Y,Z),dJ_dx2(theta0,X,Y,Z),dJ_dy2(theta0,X,Y,Z),dJ_ds2(theta0,X,Y,Z)])
        theta0 = theta0 - alpha*dJ
        niter += 1
        res.append(theta0)
        cost.append(J_cost(theta0,X,Y,Z))
    
    print('nombre d iterations :',niter)
    print(J_cost(theta0,X,Y,Z))
    res = np.array(res)
    
    return theta0,res,cost

#%% données uniformes

theta = np.array([3.7,3.7,2.5,6.3,6.3,2.5])

geometry = [10,10,31,31,0,0]
grid = init_grid(geometry)
X_uni,Y_uni = coordinate(grid)
Z_uni = f_param(theta,X_uni,Y_uni)
# bruit = np.random.normal(0,0.5,Z_uni.size)
# Z_uni = Z_uni + bruit

plt.scatter(X_uni,Y_uni,c=Z_uni,cmap='jet')
plt.axis('square')
plt.colorbar()
plt.title("données d'apprentissage uniformes")

# algorithme du gradient

alpha = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
theta0 = np.array([2,2,3,8,8,1])
Nmax = 25000


leg = ['x1','y1','s1','x2','y2','s2']

thetaf_uni,res_uni,cost_uni = estimation(X_uni,Y_uni,Z_uni,theta0,alpha,Nmax)


_,ax = plt.subplots(2,3,figsize=(10,5))
for i in range(2):
    for j in range(3):
        ax[i,j].plot(res_uni[:,3*i+j])
        ax[i,j].legend(leg[3*i+j])
        ax[i,j].axhline(y=theta[3*i+j],c='r',ls='--')
        
#%% dataset raffiné
theta = np.array([3.7,3.7,2.5,6.3,6.3,2.5])

geometry = [10,10,15,15,0,0]
grid = init_grid(geometry)
X_nu,Y_nu = coordinate(grid)
Z_nu = f_param(theta,X_nu,Y_nu)


for _ in range(2):
    grid = iterate_grid(grid,Z_nu,True)
    X_nu,Y_nu = coordinate(grid)
    Z_nu = f_param(theta,X_nu,Y_nu)
    
# bruit = np.random.normal(0,0.5,Z_nu.size)
# Z_nu = Z_nu + bruit

plt.scatter(X_nu,Y_nu,c=Z_nu,cmap='jet')
plt.axis('square')
plt.colorbar()
plt.title("données d'apprentissage raffinées")

thetaf_nu,res_nu,cost_nu = estimation(X_nu,Y_nu,Z_nu,theta0,alpha,Nmax)


_,ax = plt.subplots(2,3,figsize=(10,5))
for i in range(2):
    for j in range(3):
        ax[i,j].plot(res_nu[:,3*i+j])
        ax[i,j].legend(leg[3*i+j])
        ax[i,j].axhline(y=theta[3*i+j],c='r',ls='--')
        
#%% comparaison

_,ax = plt.subplots(2,3,figsize=(15,10))
for i in range(2):
    for j in range(3):
        ax[i,j].plot(res_uni[:,3*i+j],c='blue',label='uniforme')
        ax[i,j].plot(res_nu[:,3*i+j],c='green',label='raffiné')
        ax[i,j].set_title(leg[3*i+j])
        ax[i,j].legend()
        ax[i,j].axhline(y=theta[3*i+j],c='r',ls='--')
        
plt.figure()
plt.plot(cost_uni,label='uniforme',c='blue')
plt.plot(cost_nu,label='raffiné',c='green')
plt.legend()

plt.title('Evolution de la fonction de cout')

    
    
    