# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:29:54 2022

@author: pchauris
"""

import numpy as np
import matplotlib.pyplot as plt


from sklearn.neural_network import MLPRegressor
from sklearn import metrics


def f(X):
    return X*np.exp(-X**2)

def reg(trainX,testX):
    # train set
    trainX = trainX.reshape(-1,1)
    testX = testX.reshape(-1,1)

    trainY = f(trainX)
    
    # fit the data
    mlp_reg.fit(trainX, trainY)
    
    # predict the output
    y_pred = mlp_reg.predict(testX)
    
    print('--------------------------------------')
    print('Mean Absolute Error:', metrics.mean_absolute_error(valY, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(valY, y_pred))  
    print('--------------------------------------')
    
    return y_pred


# save the correct solution

N = 200
x = np.linspace(0,5,N)
y = f(x)

# define the MPL

mlp_reg = MLPRegressor(hidden_layer_sizes=(100,100,70),
                      max_iter = 300,activation = 'relu',
                      solver = 'adam')

# test set
Ntest = 14
testX = np.linspace(0,5,Ntest) 

# validation set

valX = np.linspace(0,5,Ntest)
valY = f(valX)

# uniform grid
Ntrain = 40
trainX_uni = np.linspace(0,5,Ntrain)

y_uni = reg(trainX_uni,testX)

# non uniform grid
trainX_nu = np.r_[np.linspace(0, 0.5, 6, endpoint=False), np.linspace(0.5, 1.2, 14, endpoint=False),
             np.linspace(1.2, 1.9, 4, endpoint=False), np.linspace(1.9, 2.9, 10, endpoint=False),
             np.linspace(2.9, 5, 6)]

y_nu = reg(trainX_nu,testX)

# results
im,(ax1,ax2) = plt.subplots(2,1)
ax1.plot(x,y)
ax1.plot(trainX_uni,f(trainX_uni),'o')
ax1.plot(valX,valY,'o')
ax1.plot(testX,y_uni,'-o')
ax1.legend(["correct","train set","validation set","prediction"])

ax2.plot(x,y)
ax2.plot(trainX_nu,f(trainX_nu),'o')
ax2.plot(valX,valY,'o')
ax2.plot(testX,y_nu,'-o')
ax2.legend(["correct","train set","validation set","prediction"])

# estimation errors
err_uni = valY - y_uni
err_nu = valY - y_nu

# plt.figure()
# plt.plot(valX,err_uni,'-o')
# plt.plot(valX,err_nu,'-o')
# plt.legend(["error with uniform grid","error with non uniform grid"])

