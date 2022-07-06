# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:46:13 2022

@author: pchauris
"""

import numpy as np
import matplotlib.pyplot as plt


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def T_direct(X):
    x1,x2 = X[:,0],X[:,1]
    # x1,x2 = X
    s = x1**2+x2**2+1
    z = np.array([s,x1/s,x2/s])
    return np.transpose(z)

def T_inverse(Z):
    z1,z2,z3 = Z[:,0],Z[:,1],Z[:,2]
    x = np.array([z1*z2,z1*z3])
    return np.transpose(x)

#%% estimation T
#------------------------------------------------------
N = 20
x = np.linspace(-1,1,N)
X1,X2 = np.meshgrid(x,x)
X = np.stack((X2,X1),-1)
X = X.reshape(N*N,2)

Y = T_direct(X)

# fit
mlp_reg = MLPRegressor(hidden_layer_sizes=(150,120,80),
                       max_iter = 300,activation = 'relu',
                       solver = 'adam')

mlp_reg.fit(X, Y)


# prediction 
Nt = 20
x = np.linspace(-1,1,Nt)
X1,X2 = np.meshgrid(x,x)
Xt = np.stack((X2,X1),-1)
Xt = Xt.reshape(Nt*Nt,2)
y_exact = T_direct(Xt)
y_pred = mlp_reg.predict(Xt)

im,(ax1,ax2) = plt.subplots(2,1)
ax1.imshow(y_exact.reshape(Nt,Nt,3)[:,:,1])
ax2.imshow(y_pred.reshape(Nt,Nt,3)[:,:,1])

print('Mean Absolute Error:', metrics.mean_absolute_error(y_exact, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_exact, y_pred))  

#%% estimation T*
#---------------------------------------------------------

mlp_reg2 = MLPRegressor(hidden_layer_sizes=(100,100,50),
                       max_iter = 300,activation = 'relu',
                       solver = 'adam')

mlp_reg2.fit(Y,X)
z = np.linspace(0,1,Nt)
Z1,Z2,Z3 = np.meshgrid(z,z,z)
Z = np.stack((Z1,Z2,Z3),-1)
Z = Z.reshape(Nt*Nt*Nt,3)
x_pred = mlp_reg2.predict(Z).reshape(Nt,Nt,Nt,2)
X_calc = T_inverse(Z).reshape(Nt,Nt,Nt,2)
