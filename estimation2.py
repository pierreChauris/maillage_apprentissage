# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:30:49 2022

@author: pierre chauris
"""

import numpy as np
import matplotlib.pyplot as plt



def f(theta,x):
    [a,b] = theta
    return a*x*np.exp(-b*x**2)

def dJ_da(theta,x):
    X = -(Y_data-f(theta,x))*x*np.exp(-b*x**2)
    return np.sum(X)/N

def dJ_db(theta,x):
    X = (Y_data-f(theta,x))*a*x**3*np.exp(-b*x**2)
    return np.sum(X)/N

# generate the data

a = 1
b = 1
theta = [a,b]
N = 50
x_uni = np.linspace(-5,5,N)
f_uni = f(theta,x_uni)
#bruit = np.random.normal(0,0.2,N)
bruit = np.array([-0.04462941,  0.04133304, -0.12484282,  0.11365125, -0.37967774,
       -0.23689244, -0.04423668,  0.1297325 , -0.00937049,  0.14260683,
       -0.06965246,  0.11816932,  0.05572795,  0.10465766,  0.20521683,
        0.16315702, -0.2433381 ,  0.45880795, -0.05389907, -0.09326108,
       -0.28345893, -0.14161946, -0.14797975, -0.1903044 , -0.13073584,
        0.12639395,  0.15959696, -0.13316231,  0.28786889,  0.14702441,
        0.02919963,  0.11225376,  0.02157862,  0.24167026,  0.02508687,
        0.24392963, -0.02571978, -0.20950212,  0.26378531, -0.18266231,
       -0.02666276, -0.10868961,  0.1944365 ,  0.25757576,  0.17381231,
       -0.05203679,  0.14356724, -0.1242826 ,  0.02300693, -0.25252006])
Y_data = f_uni + bruit

plt.scatter(x_uni,Y_data,10)
plt.plot(x_uni,f(theta,x_uni))

# descente de gradient

alpha = 0.1
theta0 = np.array([1.3,0.2])
J = np.array([dJ_da(theta0,x_uni),dJ_db(theta0,x_uni)])

Coef1 = [theta0[0]]
Coef2 = [theta0[1]]
Cost = [np.linalg.norm(J)]
niter = 0
while(np.linalg.norm(J)>0.0001):
    niter += 1
    theta0 = theta0 - alpha*J
    J = np.array([dJ_da(theta0,x_uni),dJ_db(theta0,x_uni)])
        
    Coef1.append(theta0[0])
    Coef2.append(theta0[1])
    Cost.append(np.linalg.norm(J))
    
print('nombre d iterations grille uniforme :',niter)

# affichage résultat

im,(ax1,ax2,ax3) = plt.subplots(3,1)
ax1.scatter(x_uni,Y_data,10,'b')
ax1.plot(x_uni,f(theta0,x_uni),'r')
ax1.plot(x_uni,f(theta,x_uni),'b')

ax2.plot(Coef1)
ax2.plot(Coef2)


ax3.plot(Cost)