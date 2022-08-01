# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:52:49 2022

@author: pchauris
"""

from raffinement import *
import matplotlib.pyplot as plt

def f_test(X,Y):
    return np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2)
    return np.exp(-15*(Y-0.5)**2)*np.cos(6*X)
    return np.exp(-15*(X-0.5)**2)*np.cos(6*Y)

#%% test du raffinement
geometry = [2*np.pi,1,30,10,0,0]
# geometry = [1,2*np.pi,10,30,0,0]
geometry = [10,10,20,20,0,0]
grid = init_grid(geometry)

X1,X2 = coordinate(grid)

plt.figure()
plt.scatter(X1,X2,c = f_test(X1,X2),s = 1,cmap = 'jet')
plt.colorbar()
if geometry[0]==geometry[1]:
    plt.axis('square')
    
niter = 3

for _ in range(niter):
    X1,X2 = coordinate(grid)
    Z = f_test(X1,X2)
    print('taille de la grille :',len(grid))
    grid,alpha1,alpha2,alpha3,d1,d2,d3 = iterate_grid(grid,Z,True)
    
print('taille de la grille :',len(grid))
X1,X2 = coordinate(grid)

plt.figure()
plt.scatter(X1,X2,c = f_test(X1,X2),s = 1,cmap = 'jet')
plt.title('grille raffinée')
plt.colorbar()
if geometry[0]==geometry[1]:
    plt.axis('square')
#%% schema raffinement
x = [1,1,1,1,1,1.5,1.5,1.5,1.5,1.5,2,2,2,2,2,2.5,2.5,2.5,2.5,2.5,3,3,3,3,3]
y = [1,1.5,2,2.5,3,1,1.5,2,2.5,3,1,1.5,2,2.5,3,1,1.5,2,2.5,3,1,1.5,2,2.5,3]
col1 = ['w','w','w','w','w','w','w','w','w','w','w','w','r','w','w','w','w','w','w','w','w','w','w','w','w']
col2 = ['w','w','w','w','w','w','r','w','r','w','w','w','w','w','w','w','r','w','r','w','w','w','w','w','w']
col3 = ['w','w','w','w','w','w','w','r','w','w','w','w','w','w','w','w','w','r','w','w','w','w','w','w','w']
col4 = ['w','w','w','w','w','w','w','w','w','w','w','r','w','r','w','w','w','w','w','w','w','w','w','w','w']

_,(ax1,ax2,ax3,ax4) = plt.subplots(1,4)
ax1.scatter(x,y,c=col1)
ax1.axhline(y=1,color='black',linewidth=1)
ax1.axhline(y=3,color='black',linewidth=1)
ax1.axvline(x=1,color='black',linewidth=1)
ax1.axvline(x=3,color='black',linewidth=1)
ax1.set_title(r"Cellule initiale",fontsize=10)
ax1.axis('square')
ax1.axis('off')

ax2.scatter(x,y,c=col2)
ax2.axhline(y=1,color='black',linewidth=1)
ax2.axhline(y=3,color='black',linewidth=1)
ax2.axvline(x=1,color='black',linewidth=1)
ax2.axvline(x=3,color='black',linewidth=1)
ax2.axvline(x=2,color='black',linewidth=1)
ax2.axhline(y=2,color='black',linewidth=1)
ax2.set_title(r"$S_k=\| \nabla T\|>\alpha_1$",fontsize=10)
ax2.axis('square')
ax2.axis('off')


ax3.scatter(x,y,c=col3)
ax3.axis('square')
ax3.axhline(y=1,color='black',linewidth=1)
ax3.axhline(y=3,color='black',linewidth=1)
ax3.axvline(x=1,color='black',linewidth=1)
ax3.axvline(x=3,color='black',linewidth=1)
ax3.axvline(x=2,color='black',linewidth=1)
ax3.set_title(r"$S_k= \| \frac{\partial T}{\partial x_1}\|>\alpha_2$",fontsize=10)
ax3.axis('off')

ax4.scatter(x,y,c=col4)
ax4.axhline(y=1,color='black',linewidth=1)
ax4.axhline(y=3,color='black',linewidth=1)
ax4.axvline(x=1,color='black',linewidth=1)
ax4.axvline(x=3,color='black',linewidth=1)
ax4.axhline(y=2,color='black',linewidth=1)
ax4.set_title(r"$S_k= \| \frac{\partial T}{\partial x_2}\|>\alpha_3$",fontsize=10)
ax4.axis('square')
ax4.axis('off')

#%% valeur seuil pour le raffinement
a1 = alpha1[np.where(alpha1*d1 == np.max(alpha1*d1))[0][0]]
a2 = alpha2[np.where(alpha2*d2 == np.max(alpha2*d2))[0][0]]
a3 = alpha3[np.where(alpha3*d3 == np.max(alpha3*d3))[0][0]]

_,(ax1,ax2,ax3) = plt.subplots(1,3,figsize = (15,5))
ax1.plot(alpha1,alpha1*d1)
ax1.plot(alpha1,d1)
ax1.axvline(x=a1,linestyle='--')
ax1.title.set_text(r"Seuil pour le critère $S_k=\| \nabla T\|$")
ax1.legend([r'$f(\alpha)$',r'$d(\alpha)$',r"$\alpha_{seuil}$"])
ax2.plot(alpha2,alpha2*d2)
ax2.plot(alpha2,d2)
ax2.axvline(x=a2,linestyle='--')
ax2.title.set_text(r"Seuil pour le critère $S_k= \| \frac{\partial T}{\partial x_1}\|$")
ax2.legend([r'$f(\alpha)$',r'$d(\alpha)$',r"$\alpha_{seuil}$"])
ax3.plot(alpha3,alpha3*d3)
ax3.plot(alpha3,d3)
ax3.axvline(x=a3,linestyle='--')
ax3.title.set_text(r"Seuil pour le critère $S_k= \| \frac{\partial T}{\partial x_1}\|$")
ax3.legend([r'$f(\alpha)$',r'$d(\alpha)$',r"$\alpha_{seuil}$"])

#%% comparaison du gradient

geometry = [10,10,50,50,0,0]
grid = init_grid(geometry)
X,Y = coordinate(grid)
Z = f_test(X,Y)

grad = []
exact_grad = []
nx,ny = 10,10
Coeffs = coeffs(grid,Z,nx,ny)
for cell in grid:
    gx,gy = gradient(cell,nx,ny,Coeffs)
    grad.append(np.sqrt(gx**2+gy**2))
    gx,gy = emp_grad(cell,f_test)
    exact_grad.append(np.sqrt(gx**2+gy**2))

erreur = np.linalg.norm(np.array(grad) - np.array(exact_grad))
print('erreur :',erreur)

plt.figure()
plt.scatter(X,Y,c = grad,cmap = 'jet',s=6)
plt.colorbar()
plt.title('gradient surrogate model de degré 4')
if geometry[0]==geometry[1]:
    plt.axis('square')

plt.figure()
plt.scatter(X,Y,c = exact_grad,cmap = 'jet',s=6)
plt.colorbar()
plt.title('norme du gradient')
if geometry[0]==geometry[1]:
    plt.axis('square')

#%% test R2 -> R3

def f2d(X,Y):
    return np.stack((np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2),Y*np.sin(X)),-1)

geometry1 = [10,10,10,10,0,0]
geometry2 = [2*np.pi,1,10,10,0,0]
grid1 = init_grid(geometry1)
grid2 = init_grid(geometry2)
X,Y = coordinate(grid1)
Z = f2d(X,Y)

for i in range(2):
    plt.figure()
    plt.scatter(X,Y,c = Z[:,i],s=1)
    
sup_grid = [grid1,grid2]
# iteration de raffinement
for i in range(len(sup_grid)):
    sup_grid[i] = iterate_grid(sup_grid[i],Z[:,i],True)

for gr in sup_grid:
    x,y = coordinate(gr)
    plt.figure()
    plt.scatter(x,y,s=1)
# calcul des nouveaux Z
x,y = coordinate(sup_grid[0])
Z1 = f2d(x,y)[:,0]
x,y = coordinate(sup_grid[1])
Z2 = f2d(x,y)[:,1]
# iteration de raffinement
sup_grid[0] = iterate_grid(sup_grid[0],Z1,True)
sup_grid[1] = iterate_grid(sup_grid[1],Z2,True)

# calcul des nouveaux Z
x,y = coordinate(sup_grid[0])
Z1 = f2d(x,y)[:,0]
plt.figure()
plt.scatter(x,y,c=Z1,s=1)

x,y = coordinate(sup_grid[1])
Z2 = f2d(x,y)[:,1]
plt.figure()
plt.scatter(x,y,c=Z2,s=1)

# iteration de raffinement
sup_grid[0] = iterate_grid(sup_grid[0],Z1,False)
sup_grid[1] = iterate_grid(sup_grid[1],Z2,False)


for gr in sup_grid:
    x,y = coordinate(gr)
    plt.figure()
    plt.scatter(x,y,s=1)
