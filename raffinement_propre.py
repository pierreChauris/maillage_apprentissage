# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:11:27 2022

@author: pchauris
"""

import numpy as np
import matplotlib.pyplot as plt
import time
#%% 
def f(X,Y):
    return np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2)
    return Y*np.sin(X)


#%% fonctions
class Cell:
    def __init__(self,index,geometry):
        
        Lx, Ly, Nx, Ny, Ox, Oy = geometry
        px,py = Lx/Nx,Ly/Ny
        x = index[0]*px + px/2 + Ox
        y = index[1]*py + py/2 + Oy
        
        self.index = index
        self.geometry = geometry
        self.center = np.array([x,y])
        self.size = np.array([px,py])
        
    def info(self):
        print('index :',self.index)
        print('center :',self.center)
        print('size :',self.size)
        
    def split_iso(self):
        px,py = self.size
        #create 4 new cells
        C00 = Cell(np.concatenate((self.index,[0,0])),self.geometry)
        C00.size = self.size/2
        C00.center = self.center - np.array([px/4,py/4])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),self.geometry)
        C01.size = self.size/2
        C01.center = self.center + np.array([px/4,-py/4])
        
        C10 = Cell(np.concatenate((self.index,[1,0])),self.geometry)
        C10.size = self.size/2
        C10.center = self.center + np.array([-px/4,py/4])
        
        C11 = Cell(np.concatenate((self.index,[1,1])),self.geometry)
        C11.size = self.size/2
        C11.center = self.center + np.array([px/4,py/4])
        
        return C00,C01,C10,C11
    
    def split_x(self):
        px,py = self.size
        #create 2 new cells allong first axis
        C00 = Cell(np.concatenate((self.index,[0,0])),self.geometry)
        C00.size = np.array([px/2,py])
        C00.center = self.center - np.array([px/4,0])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),self.geometry)
        C01.size = np.array([px/2,py])
        C01.center = self.center + np.array([px/4,0])
        
        return C00,C01
    
    
    def split_y(self):
        px,py = self.size
        #create 2 new cells allong second axis
        C00 = Cell(np.concatenate((self.index,[0,0])),self.geometry)
        C00.size = np.array([px,py/2])
        C00.center = self.center - np.array([0,py/4])
        
        C01 = Cell(np.concatenate((self.index,[0,1])),self.geometry)
        C01.size = np.array([px,py/2])
        C01.center = self.center + np.array([0,py/4])
        
        return C00,C01

def init_grid(geometry):
    grid = []
    Nx,Ny = geometry[2:4]
    for i in range(Nx):
        for j in range(Ny):
            cell = Cell(np.array([i,j]),geometry)
            grid.append(cell)
    return grid


def coordinate(grid):
    X1 = []
    X2 = []
    for cell in grid:
        x1,x2 = cell.center
        X1.append(x1)
        X2.append(x2)
    return np.array(X1),np.array(X2)


def emp_grad(cell):
    x1,y1 = cell.center
    dx,dy = cell.size
    x0,y0 = x1 - dx/2 , y1 - dy/2
    Dfx = f(x0+dx,y1) - f(x0,y1)
    Dfy = f(x1,y0+dy) - f(x1,y0)
    return np.abs(Dfx/dx),np.abs(Dfy/dy)


def crit_sequence(grid,Z,nx,ny,Coeffs):
    iso,vert,horiz = [],[],[]
    for cell in grid:
        gx,gy = gradient(cell,nx,ny,Coeffs,4)
        iso.append(np.sqrt(gx**2+gy**2))
        vert.append(np.abs(gx))
        horiz.append(np.abs(gy))
    return np.array(iso),np.array(vert),np.array(horiz)

def alpha_sequence(grid,Z,nx,ny,Coeffs):
    res1,res2,res3 = crit_sequence(grid,Z,nx,ny,Coeffs)
    return np.linspace(0,res1.mean(),len(grid)),np.linspace(0,res2.mean(),len(grid)),np.linspace(0,res3.mean(),len(grid))

def distrib_sequence(grid,Z,nx,ny,Coeffs):
    alpha1,alpha2,alpha3 = alpha_sequence(grid,Z,nx,ny,Coeffs)
    crit1,crit2,crit3 = crit_sequence(grid,Z,nx,ny,Coeffs)
    res1,res2,res3 = [],[],[]
    for j in range(alpha1.size):
        dj1 = np.count_nonzero(crit1>alpha1[j])
        dj2 = np.count_nonzero(crit2>alpha2[j])
        dj3 = np.count_nonzero(crit3>alpha3[j])
        res1.append(dj1)
        res2.append(dj2)
        res3.append(dj3)
    return np.array(res1),np.array(res2),np.array(res3)

def auto_threshold(grid,Z,nx,ny,Coeffs):
    alpha1,alpha2,alpha3 = alpha_sequence(grid,Z,nx,ny,Coeffs)
    d1,d2,d3 = distrib_sequence(grid,Z,nx,ny,Coeffs)
    f1,f2,f3 = alpha1*d1,alpha2*d2,alpha3*d3
    #maximum global
    fmax1,fmax2,fmax3 = np.max(f1), np.max(f2), np.max(f3)
    idmax1,idmax2,idmax3 = np.where(f1 == fmax1),np.where(f2 == fmax2),np.where(f3 == fmax3)
    idmax1,idmax2,idmax3 = idmax1[0][0],idmax2[0][0],idmax3[0][0]
    #alpha 
    alphamax1,alphamax2,alphamax3 = alpha1[idmax1], alpha2[idmax2], alpha3[idmax3]
    return alphamax1,alphamax2,alphamax3

def iterate_grid(grid,Z,smooth):
    # nombre de surrogate models optimal pour des surrogates models de degré 3
    nx = int(np.sqrt(len(grid))/5)
    ny = int(np.sqrt(len(grid))/5)
    Coeffs = coeffs(grid,Z,nx,ny,4)
    alpha,alphax,alphay = auto_threshold(grid,Z,nx,ny,Coeffs)
    new_grid = grid.copy()
    
    for cell in grid:
        k = grid.index(cell)
        gx,gy = gradient(cell,nx,ny,Coeffs,4)
        
        raffinement = 0        
        
        # smooth grid
        if smooth:
            Lx,Ly = cell.geometry[0:2]
            px,py = cell.size
            px = px*Ly
            py = py*Lx
            if py > 2*px:
                C00,C01 = cell.split_y()
                if cell in new_grid:
                    new_grid.remove(cell)
                new_grid.insert(k,C01)
                new_grid.insert(k,C00)
                raffinement += 1
            
            if px > 2*py:
                C00,C01 = cell.split_x()
                if cell in new_grid:
                    new_grid.remove(cell)
                new_grid.insert(k,C01)
                new_grid.insert(k,C00)
                raffinement += 1
            
        # raffinement uniquement sur x 
        if gx > 15*gy and gx > alphax and raffinement == 0:
            C00,C01 = cell.split_x()
            if cell in new_grid:
                new_grid.remove(cell)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
            raffinement += 1
            
        # reffinement uniquement sur y 
        if gy > 15*gx and gy > alphay and raffinement == 0:
            C00,C01 = cell.split_y()
            if cell in new_grid:
                new_grid.remove(cell)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
            raffinement += 1
        
        # sinon raffinement iso
        if np.sqrt(gx**2+gy**2) > alpha and raffinement == 0:
            C00,C01,C10,C11 = cell.split_iso()
            new_grid.remove(cell)
            new_grid.insert(k,C11)
            new_grid.insert(k,C10)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
         
    return new_grid

def split_grid(grid,Z,nx,ny,ix,iy):
    "return the sub grid of index (ix,iy) from the grid divided in nx time ny regions"
    sub_grid = []
    sub_Z = []
    for cell in grid:
        ind = grid.index(cell)
        x,y = cell.center
        Lx,Ly = cell.geometry[0:2]
        Ox,Oy = cell.geometry[4:6]
        if ix*Lx/nx+Ox < x < (ix+1)*Lx/nx+Ox and iy*Ly/ny+Oy < y < (iy+1)*Ly/ny+Oy:
            sub_grid.append(cell)
            sub_Z.append(Z[ind])
    return sub_grid,sub_Z

def surr_model(sub_grid,sub_Z,degre):
    "return the array of coefficients of the polynomial model fit over sub_grid"
    X,Y = coordinate(sub_grid)
    if degre == 1:
        A = np.stack((X,Y,np.ones(X.size)),-1)
    if degre == 2:
        A = np.stack((X*X,X*Y,Y*Y,X,Y,np.ones(X.size)),-1)
    if degre == 3:
        A = np.stack((X*X*X,X*X*Y,X*Y*Y,Y*Y*Y,X*X,X*Y,Y*Y,X,Y,np.ones(X.size)),-1)
    if degre == 4:
        A = np.stack((X**4,Y*X**3,(X*Y)**2,X*Y**3,Y**4,X*X*X,X*X*Y,X*Y*Y,Y*Y*Y,X*X,X*Y,Y*Y,X,Y,np.ones(X.size)),-1)
    A_star = np.linalg.pinv(A)
    param = np.dot(A_star,sub_Z)
    return param

def coeffs(grid,Z,nx,ny,degre):
    "return the list of coefficients of all the nx time ny surrogate models of the grid"
    Coeffs = []
    for iy in range(ny):
        for ix in range(nx):
            sub_grid,sub_Z = split_grid(grid,Z,nx,ny,ix,iy)
            param = surr_model(sub_grid,sub_Z,degre)
            Coeffs.append(param)
    return Coeffs

def gradient(cell,nx,ny,Coeffs,degre):
    "compute the gradient of the cell from the surrogate model where the cell is located"
    x,y = cell.center
    Lx,Ly = cell.geometry[0:2]
    Ox,Oy = cell.geometry[4:6]
    ix,iy = int((x-Ox)/(Lx/nx)), int((y-Oy)/(Ly/ny))
    if degre == 1:
        [a,b,c] = Coeffs[iy*nx+ix]
        gx = a
        gy = b
    if degre == 2:
        [a,b,c,d,e,f] = Coeffs[iy*nx+ix]
        gx = 2*a*x + b*y + d
        gy = 2*c*y + b*x + e
    if degre == 3:
        [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10] = Coeffs[iy*nx+ix]
        gx = 3*a1*x**2 + 2*a2*x*y + a3*y**2 + 2*a5*x + a6*y + a8
        gy = a2*x**2 + 2*a3*x*y + 3*a4*y**2 + a6*x + 2*a7*y + a9
    if degre == 4:
        [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15] = Coeffs[iy*nx+ix]
        gx = 4*a1*x**3 + 3*a2*x**2*y + 2*a3*x*y**2 + a4*y**3 + 3*a6*x**2 + 2*a7*x*y + a8*y**2 + 2*a10*x + a11*y + a13
        gy = a2*x**3 + 2*a3*x**2*y + 3*a4*x*y**2 + 4*a5*y**3 + a7*x**2 + 2*a8*x*y + 3*a9*y**2 + a11*x + 2*a12*y + a14
    return abs(gx),abs(gy)

#%%
start = time.time()
geometry = [2*np.pi,1,20,11,0,0]
geometry = [10,10,30,30,0,0]
grid = init_grid(geometry)

niter = 2

for _ in range(niter):
    X1,X2 = coordinate(grid)
    Z = f(X1,X2)
    print('taille de la grille :',len(grid))
    grid = iterate_grid(grid,Z,True)
    
print('taille de la grille :',len(grid))
X1,X2 = coordinate(grid)

plt.figure()
plt.scatter(X1,X2,c = f(X1,X2),s = 1,cmap = 'jet')
plt.colorbar()
if geometry[0]==geometry[1]:
    plt.axis('square')
end = time.time() - start
print(end)

#%%

grid = init_grid([10,10,20,20,0,0])
X,Y = coordinate(grid)
Z = f(X,Y)


nx,ny = 3,3

sub_grid,sub_Z = split_grid(grid,Z,nx,ny,1,2)

plt.scatter(X,Y,c = 'white')
plt.axis('square')

X0,Y0 = coordinate(sub_grid)
plt.scatter(X0,Y0,c = sub_Z)
plt.axis('square')
#%%
import random
nx,ny = 10,10
col = np.array(['r','g','b','y'])
plt.scatter(X,Y,c = 'white')
plt.axis('square')
for ix in range(nx):
    for iy in range(ny):
        sub_grid,sub_Z = split_grid(grid,Z,nx,ny,ix,iy)
        X0,Y0 = coordinate(sub_grid)
        plt.scatter(X0,Y0,c = col[random.randint(0,3)],s=10)
        plt.axis('square')
        
    

#%%
grid = init_grid([10,10,40,40,0,0])
X,Y = coordinate(grid)
Z = f(X,Y)

grad = []
exact_grad = []
nx,ny = 8,8
degre = 4
Coeffs = coeffs(grid,Z,nx,ny,degre)
for cell in grid:
    gx,gy = gradient(cell,nx,ny,Coeffs,degre)
    grad.append(np.sqrt(gx**2+gy**2))
    gx,gy = emp_grad(cell)
    exact_grad.append(np.sqrt(gx**2+gy**2))

erreur = np.linalg.norm(np.array(grad) - np.array(exact_grad))
print('erreur :',erreur)

plt.figure()
plt.scatter(X,Y,c = grad,cmap = 'jet')
plt.colorbar()
plt.title('gradient surrogate model de degré 4')

plt.figure()
plt.scatter(X,Y,c = exact_grad,cmap = 'jet')
plt.colorbar()
plt.title('gradient exact')
#%%
N = 30
degre = 4
grid = init_grid([10,10,N,N,0,0])
X,Y = coordinate(grid)
Z = f(X,Y)
err = []
for i in range(4,15):
    print(i)
    grad = []
    exact_grad = []
    nx,ny = i,i
    Coeffs = coeffs(grid,Z,nx,ny,degre)
    for cell in grid:
        gx,gy = gradient(cell,nx,ny,Coeffs,degre)
        grad.append(np.sqrt(gx**2+gy**2))
        gx,gy = emp_grad(cell)
        exact_grad.append(np.sqrt(gx**2+gy**2))

    erreur = np.linalg.norm(np.array(grad) - np.array(exact_grad))
    err.append(erreur)

#%%
err_d1 = err
#%%
err_d2 = err
#%%
err_d3 = err
#%%
err_d4 = err
#%%
plt.title('erreur sur le gradient en fonction du nombre de surrogate models N = 40')
plt.plot([i for i in range(4,15)],err_d2,'-o',color='blue')
plt.plot([i for i in range(4,15)],err_d3,'-o',color='orange')
plt.plot([i for i in range(4,15)],err_d4,'-o',color='green')
plt.axvline(x = N//5,linestyle='--',color = 'green')
plt.axvline(x = N//4,linestyle='--',color = 'orange')
plt.axvline(x = N//3,linestyle='--',color = 'blue')

plt.legend(['degré 2','degré 3','degré 4'])
plt.xlabel('nombre de surrogate models par axe')
plt.ylabel('norme de l erreur')

#%%

def Poly(n,X,Y):
    pol = np.zeros(X.shape)
    for k in range(n+1):
        for i in range(k+1):
            pol += ((-1)**i)*(X**i)*(Y**(k-i))
    return pol

n = 5
X = np.linspace(-20,20,100)
X,Y = np.meshgrid(X,X)
pol = Poly(n,X,Y)

plt.scatter(X,Y,c=pol)

#%%
grid = init_grid([4,4,50,50,1,1])
X,Y = coordinate(grid)
Z = np.exp(-((X-3)**2 + (Y-3)**2))

plt.scatter(X,Y,c=Z,cmap='jet')
plt.colorbar()
plt.show()
#%%
grad = []
nx,ny = 10,10
Coeffs = coeffs(grid,Z,nx,ny)
for cell in grid:
    gx,gy = gradient(cell,nx,ny,Coeffs)
    grad.append(np.sqrt(gx**2+gy**2))
#%%
plt.scatter(X,Y,c = grad,s=5,cmap = 'jet')
plt.colorbar()
plt.title('gradient surrogate model de degré 3')
plt.show()