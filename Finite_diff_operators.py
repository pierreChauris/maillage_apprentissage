# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff



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

def dTinv_dz(i,j):
    A = np.vstack((dTx1[:,i,j],dTx2[:,i,j]))
    A = np.transpose(A)
    return np.linalg.pinv(A)


# define the grid and compute T(x)=Z and dT/dx on the grid-------------------------------
N = 20
x1 = np.linspace(-1,1,N)
x2 = x1
X1,X2 = np.meshgrid(x1,x2)
plt.scatter(X1,X2,1)
plt.show()

Z = T_direct(X1,X2)
dTx = dT_dx(X1,X2)

# plot the 3 components of T(x)---------------------------------------------------------
f1, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(Z[0,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax2.imshow(Z[1,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax3.imshow(Z[2,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
plt.show()

# plot the 6 components of dT/dx from analytic expression--------------------------------
f2,ax = plt.subplots(3,2)
ax[0,0].imshow(dTx[0,0,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[0,1].imshow(dTx[0,1,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[1,0].imshow(dTx[1,0,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[1,1].imshow(dTx[1,1,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[2,0].imshow(dTx[2,0,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[2,1].imshow(dTx[2,1,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])

plt.show()

# plot the 6 components of dT/dx from FinDiff computation--------------------------------
d_dx1 = FinDiff(2,1,1)
d_dx2 = FinDiff(1,1,1)
dTx1 = d_dx1(Z)
dTx2 = d_dx2(Z)


f3,ax = plt.subplots(3,2)
ax[0,0].imshow(dTx1[0,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[0,1].imshow(dTx2[0,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[1,0].imshow(dTx1[1,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[1,1].imshow(dTx2[1,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[2,0].imshow(dTx1[2,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[2,1].imshow(dTx2[2,:,:],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])

plt.show()

# compute dTinv_dz from dTinv_dz(z).dT_dx(x) = Id
dTz = []
for i in range(N):
    for j in range(N):
        dTz.append(dTinv_dz(i,j))
dTz = np.array(dTz)
dTz = np.reshape(dTz,(N,N,2,3))

f4,ax = plt.subplots(2,3)
ax[0,0].imshow(dTz[:,:,0,0],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[0,1].imshow(dTz[:,:,0,1],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[0,2].imshow(dTz[:,:,0,2],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[1,0].imshow(dTz[:,:,1,0],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[1,1].imshow(dTz[:,:,1,1],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
ax[1,2].imshow(dTz[:,:,1,2],extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)])

