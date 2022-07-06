# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:41:21 2022

@author: pierre chauris
"""

import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff

# test pour le git
# test avec le git bash fermé

def T_direct(x1,x2):
    s = x1**2+x2**2+1
    z = np.array([s,x1/s,x2/s])
    return z

def T_inverse(z1,z2,z3):
    return np.array([z1*z2,z1*z3])

def dT_dx(x1,x2):
    den = (x1**2+x2**2+1)**2
    return np.array([[2*x1,2*x2],
                     [(x2**2-x1**2+1)/den,-2*x1*x2/den],
                     [-2*x1*x2/den,(x1**2-x2**2+1)/den]])
def dT_dz(x1,x2):
    z = T_direct(x1,x2)
    z1,z2,z3 = z
    return np.array([[z2,z1,0],[z3,0,z1]])
    
def dT_dz_emp_ij(i,j):
    return np.linalg.pinv(dTdx[:,:,i,j])

def dT_dz_emp():
    dTdz_emp = np.array([dT_dz_emp_ij(i,j) for (i,j) in flat_index])
    return np.reshape(dTdz_emp,(N,N,2,3))

def Tinv_est(theta,z):
    a1,b1,c1,a2,b2,c2 = theta
    z1,z2,z3 = z
    return np.array([a1*z1*z2 + b1*z1*z3 + c1*z2*z3
                    ,a2*z1*z2 + b2*z1*z3 + c2*z2*z3])
    
def dJ_da1(theta):
    arg = np.array([[Z[0,i,j]*Z[1,i,j]*((Tinv_est(theta,Z[:,i,j]))[0]-x1[j]) for i in range(N)] for j in range(N)])
    return np.sum(arg)/N**2

def dJ_db1(theta):
    arg = np.array([[Z[0,i,j]*Z[2,i,j]*((Tinv_est(theta,Z[:,i,j]))[0]-x1[j]) for i in range(N)] for j in range(N)])
    return np.sum(arg)/N**2

def dJ_dc1(theta):
    arg = np.array([[Z[1,i,j]*Z[2,i,j]*((Tinv_est(theta,Z[:,i,j]))[0]-x1[j]) for i in range(N)] for j in range(N)])
    return np.sum(arg)/N**2

def dJ_da2(theta):
    arg = np.array([[Z[0,i,j]*Z[1,i,j]*((Tinv_est(theta,Z[:,i,j]))[1]-x1[i]) for i in range(N)] for j in range(N)])
    return np.sum(arg)/N**2

def dJ_db2(theta):
    arg = np.array([[Z[0,i,j]*Z[2,i,j]*((Tinv_est(theta,Z[:,i,j]))[1]-x1[i]) for i in range(N)] for j in range(N)])
    return np.sum(arg)/N**2

def dJ_dc2(theta):
    arg = np.array([[Z[1,i,j]*Z[2,i,j]*((Tinv_est(theta,Z[:,i,j]))[1]-x1[i]) for i in range(N)] for j in range(N)])
    return np.sum(arg)/N**2

# define the grid and compute T(x)=Z and dT/dx on the grid-------------------------------
N = 100
index = [[(i,j) for i in range(N)]for j in range(N)]
flat_index = [num for row in index for num in row]
x1 = np.linspace(-1,1,N)
x2 = x1
X1,X2 = np.meshgrid(x1,x2)

Z = T_direct(X1,X2)
dTdx = dT_dx(X1,X2)
dTdz = dT_dz(X1,X2)
dTdz_emp = dT_dz_emp()


# optimize the parameters of Tinv_est

alpha = 0.1
theta0 = np.array([0,1,0.1,1,0,0.1])
J = np.array([dJ_da1(theta0),dJ_db1(theta0),dJ_dc1(theta0),dJ_da2(theta0),dJ_db2(theta0),dJ_dc2(theta0)])

a1 = [theta0[0]]
b1 = [theta0[1]]
c1 = [theta0[2]]
a2 = [theta0[3]]
b2 = [theta0[4]]
c2 = [theta0[5]]
Cost = [np.linalg.norm(J)]

niter = 0
while(np.linalg.norm(J)>0.0001):
    print(np.linalg.norm(J))
    niter += 1
    theta0 = theta0 - alpha*J
    J = np.array([dJ_da1(theta0),dJ_db1(theta0),dJ_dc1(theta0),dJ_da2(theta0),dJ_db2(theta0),dJ_dc2(theta0)])
    a1.append(theta0[0])
    b1.append(theta0[1])
    c1.append(theta0[2])
    a2.append(theta0[3])
    b2.append(theta0[4])
    c2.append(theta0[5])
    Cost.append(np.linalg.norm(J))
    
i,j = [10,56]
x_true = np.array([x1[i],x2[j]])
x_est = Tinv_est(theta0,T_direct(x_true[0],x_true[1]))
print('theta :',theta0)
print('x_true :',x_true)
print('x_est :',x_est)

img,ax = plt.subplots(2,3)
ax[0,0].plot(a1)
ax[0,1].plot(b1)
ax[0,2].plot(c1)
ax[1,0].plot(a2)
ax[1,1].plot(b2)
ax[1,2].plot(c2)
