# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:58:43 2022

@author: pierre chauris
"""

import numpy as np
import matplotlib.pyplot as plt



def f(theta,x):
    [a,b] = theta
    return a*x + b

def dJ_da(theta,k):
    X = (Y_data-f(theta,x_uni))*x_uni**k
    return np.sum(X)/N

# generate the data

a = 1
b = 0.5
theta = [a,b]
N = 30
x_uni = np.linspace(-5,5,N)
f_uni = f(theta,x_uni)
Y_data = f_uni + np.random.normal(1,1,N)

plt.scatter(x_uni,Y_data)
plt.plot(x_uni,f(theta,x_uni))

# algorithme du gradient

alpha = 0.01
theta0 = np.array([0,0])
n_para = theta0.size
J = np.array([dJ_da(theta0,n_para-1-k) for k in range(n_para)])

Coef1 = [theta0[0]]
Coef2 = [theta0[1]]
Cost = [np.linalg.norm(J)]

while(np.linalg.norm(J)>0.01):
    theta0 = theta0 + alpha*J
    J = np.array([dJ_da(theta0,n_para-1-k) for k in range(n_para)])
    
    print(np.linalg.norm(J))
    
    Coef1.append(theta0[0])
    Coef2.append(theta0[1])
    Cost.append(np.linalg.norm(J))

    

# affichage résultat
im,(ax1,ax2,ax3) = plt.subplots(3,1)
ax1.scatter(x_uni,Y_data)
ax1.plot(x_uni,f(theta0,x_uni))

ax2.plot(Coef1)
ax2.plot(Coef2)

ax3.plot(Cost)