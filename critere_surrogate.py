# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:47:08 2022

@author: pchauris
"""

"Objectif : définir et tester un nouveau critère basé sur l'erreur d'interpolation par surrogate model"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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

def plot_cell(cell):
    x,y = cell.center
    px,py = cell.size
    plt.vlines(x = x-px/2, ymin = y-py/2, ymax = y+py/2,linewidth=1)
    plt.vlines(x = x+px/2, ymin = y-py/2, ymax = y+py/2,linewidth=1)
    plt.hlines(y = y-py/2, xmin = x-px/2, xmax = x+px/2,linewidth=1)
    plt.hlines(y = y+py/2, xmin = x-px/2, xmax = x+px/2,linewidth=1)
    
def erreur_surrogate(grid):
    X,Y = coordinate(grid)
    Lx,Ly,Nx,Ny,Ox,Oy = grid[0].geometry
    grid0 = init_grid([Lx,Ly,Nx//4,Ny//4,Ox,Oy])
    X0,Y0 = coordinate(grid0)
    Z_estime = interpole_grid(grid0,f(X0,Y0),grid)
    erreur = np.abs(Z_estime - f(X,Y))
    return erreur

def crit_sequence(grid,method):
    if method == 'gradient':
        return grad(grid)
    if method == 'erreur':
        return erreur_surrogate(grid)

def alpha_sequence(grid,method):
    sequence = crit_sequence(grid,method)
    return np.linspace(0,sequence.max(),len(grid))


def distrib_sequence(grid,method):
    alpha = alpha_sequence(grid,method)
    crit = crit_sequence(grid,method)
    distribution = []
    for j in range(alpha.size):
        dj = np.count_nonzero(crit>alpha[j])
        distribution.append(dj)
    return np.array(distribution)


def auto_threshold(grid,method):
    alpha = alpha_sequence(grid,method)
    distribution = distrib_sequence(grid,method)
    f = alpha*distribution
    #maximum global
    fmax = np.max(f)
    idmax = np.where(f == fmax)
    idmax = idmax[0][0]
    #alpha 
    alphamax = alpha[idmax]
    return alphamax

def iterate_grid(grid,method):
    alpha = auto_threshold(grid,method)
    crit = crit_sequence(grid,method)
    new_grid = grid.copy()
    for cell in grid:
        k = grid.index(cell)        
        # raffinement iso
        if crit[k] > alpha :
            C00,C01,C10,C11 = cell.split_iso()
            new_grid.remove(cell)
            new_grid.insert(k,C11)
            new_grid.insert(k,C10)
            new_grid.insert(k,C01)
            new_grid.insert(k,C00)
        # if len(new_grid) > 768:
        #     break   
    return new_grid


def uniform_grid(dimensions,npts):
    Lx,Ly,Nx,Ny,Ox,Oy = dimensions
    N = int(np.sqrt(npts)) + 1
    x,y = np.linspace(Ox,Ox+Lx,N), np.linspace(Oy,Oy+Ly,N)
    X,Y = np.meshgrid(x,y)
    while X.size != npts:
        ind = np.random.randint(0,X.size)
        X = np.delete(X,ind)
        Y = np.delete(Y,ind)
    return X.flatten(),Y.flatten()


def split_grid(grid,Z,nx,ny,ix,iy):
    "calcule le masque correspondant au surrogate ix iy et retourne les données du surrogate"
    X,Y = coordinate(grid)
    Lx,Ly = grid[0].geometry[0:2]
    Ox,Oy = grid[0].geometry[4:6]
    Ax = (ix*Lx/nx+Ox)*np.ones(X.size)
    Bx = ((ix+1)*Lx/nx+Ox)*np.ones(X.size)
    Ay = (iy*Ly/ny+Oy)*np.ones(Y.size)
    By = ((iy+1)*Ly/ny+Oy)*np.ones(Y.size)
    
    mask = (np.less(Ax,X) & np.less(X,Bx)) & (np.less(Ay,Y) & np.less(Y,By))
    mask = ~mask
    return np.ma.masked_array(X,mask).compressed(),np.ma.masked_array(Y,mask).compressed(),np.ma.masked_array(Z,mask).compressed()

def surr_model(X,Y,Z):
    "return the array of coefficients of the polynomial model P such that Z = P(X,Y)"
    # degré 4
    A = np.stack((X**4,Y*X**3,(X*Y)**2,X*Y**3,Y**4,X*X*X,X*X*Y,X*Y*Y,Y*Y*Y,X*X,X*Y,Y*Y,X,Y,np.ones(X.size)),-1)
    # resolution du système
    A_star = np.linalg.pinv(A)
    param = np.dot(A_star,Z)
    return param


def coeffs(grid,Z,nx,ny):
    "return the list of coefficients of all the nx time ny surrogate models of the grid"
    Coeffs = []
    for iy in range(ny):
        for ix in range(nx):
            subX,subY,subZ = split_grid(grid,Z,nx,ny,ix,iy)
            param = surr_model(subX,subY,subZ)
            Coeffs.append(param)
    return Coeffs

def interpolation(cell,nx,ny,Coeffs):
    x,y = cell.center
    Lx,Ly = cell.geometry[0:2]
    Ox,Oy = cell.geometry[4:6]
    ix,iy = int((x-Ox)/(Lx/nx)), int((y-Oy)/(Ly/ny))
    # degré 4
    [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15] = Coeffs[iy*nx+ix]
    z_estime = a1*x**4 + a2*y*x**3 + a3*(x*y)**2 + a4*x*y**3 + a5*y**4 + a6*x**3 + a7*y*x**2 + a8*x*y**2 + a9*y**3 + a10*x**2 + a11*x*y + a12*y**2 + a13*x + a14*y + a15
    return np.array(z_estime)

def interpole_grid(grid,Z,interpolation_grid):
    Nx,Ny = grid[0].geometry[2:4]
    nx,ny = Nx//5,Ny//5
    Coeffs = coeffs(grid,Z,nx,ny)
    Z_estime = []
    for cell in interpolation_grid:
        Z_estime.append(interpolation(cell,nx,ny,Coeffs))
    return np.array(Z_estime)

def gradient(cell,nx,ny,Coeffs):
    "compute the gradient of the cell from the surrogate model where the cell is located"
    x,y = cell.center
    Lx,Ly = cell.geometry[0:2]
    Ox,Oy = cell.geometry[4:6]
    ix,iy = int((x-Ox)/(Lx/nx)), int((y-Oy)/(Ly/ny))
    # degré 4
    [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15] = Coeffs[iy*nx+ix]
    gx = 4*a1*x**3 + 3*a2*x**2*y + 2*a3*x*y**2 + a4*y**3 + 3*a6*x**2 + 2*a7*x*y + a8*y**2 + 2*a10*x + a11*y + a13
    gy = a2*x**3 + 2*a3*x**2*y + 3*a4*x*y**2 + 4*a5*y**3 + a7*x**2 + 2*a8*x*y + 3*a9*y**2 + a11*x + 2*a12*y + a14
    return abs(gx),abs(gy)

def grad(grid):
    res = []
    X,Y = coordinate(grid)
    Nx,Ny = grid[0].geometry[2:4]
    nx,ny = Nx//5,Ny//5
    Coeffs = coeffs(grid,f(X,Y),nx,ny)
    for cell in grid:
        gx,gy = gradient(cell,nx,ny,Coeffs)
        res.append(np.sqrt(gx**2+gy**2))
    return np.array(res)
#%%
def f(X,Y):
    return np.exp(-0.4*(X-3.7)**2 - 0.4*(Y-3.7)**2) + np.exp(-0.4*(X-6.3)**2 - 0.4*(Y-6.3)**2)

Nx,Ny = 40,40
geometry = [10,10,Nx,Ny,0,0]
grid = init_grid(geometry)
X,Y = coordinate(grid)
nx,ny = Nx//5,Ny//5
Coeffs = coeffs(grid,f(X,Y),nx,ny)

plt.scatter(X,Y,c=f(X,Y),cmap='jet')
plt.axis('square')
plt.colorbar()
plt.title('transformation à apprendre')

interpolation_grid = init_grid([10,10,12,12,0,0])
Z_estime = interpole_grid(grid,f(X,Y),interpolation_grid)
    
X,Y = coordinate(interpolation_grid)
Z_exact = f(X,Y)
erreur = crit_sequence(interpolation_grid,'gradient')
plt.figure()
plt.scatter(X,Y,c=erreur,cmap='jet')
plt.axis('square')
plt.colorbar()

#%%

X,Y = coordinate(grid)
erreur = crit_sequence(grid,'gradient')
plt.figure()
plt.scatter(X,Y,c=erreur,cmap='jet')
plt.axis('square')
plt.colorbar()

grid = iterate_grid(grid,'erreur')
X,Y = coordinate(grid)
plt.figure()
plt.scatter(X,Y,c=f(X,Y),cmap='jet',s=1)
plt.axis('square')
plt.colorbar()

#%%

Nx,Ny = 20,20
geometry = [10,10,Nx,Ny,0,0]
grid = init_grid(geometry)
X,Y = coordinate(grid)

erreur = crit_sequence(grid,'gradient')
plt.figure()
plt.scatter(X,Y,c=erreur,cmap='jet')
plt.axis('square')
plt.colorbar()


grid2 = iterate_grid(grid,'erreur')
X2,Y2 = coordinate(grid2)
plt.figure()
plt.scatter(X2,Y2,c=f(X2,Y2),cmap='jet',s=1)
plt.axis('square')
plt.colorbar()

grid3 = iterate_grid(grid,'gradient')
X3,Y3 = coordinate(grid3)
plt.figure()
plt.scatter(X3,Y3,c=f(X3,Y3),cmap='jet',s=1)
plt.axis('square')
plt.colorbar()

#%% données d'entrainement non uniforme 1

train_X_nu1 = np.stack((X2,Y2),-1)
sc=StandardScaler()
scaler_nu1 = sc.fit(train_X_nu1)
train_X_nu1 = scaler_nu1.transform(train_X_nu1)
train_Z_nu1 = f(X2,Y2)

#%% données d'entrainement non uniforme 2

train_X_nu2 = np.stack((X3,Y3),-1)
sc=StandardScaler()
scaler_nu2 = sc.fit(train_X_nu2)
train_X_nu2 = scaler_nu2.transform(train_X_nu2)
train_Z_nu2 = f(X3,Y3)


#%% données d'apprentissage uniformes équivalentes

Xu,Yu = uniform_grid(geometry,len(grid3))
train_X_uni = np.stack((Xu,Yu),-1)
sc=StandardScaler()
scaler_uni = sc.fit(train_X_uni)
train_X_uni = scaler_uni.transform(train_X_uni)
train_Z_uni = f(Xu,Yu)

#%% apprentissage uniforme et non uniforme

data_in = train_X_uni
data_out = train_Z_uni
# fit
mlp_reg_uni = MLPRegressor(hidden_layer_sizes=(30,20),
                       max_iter = 1000,activation = 'relu',
                       solver = 'adam')

mlp_reg_uni.fit(data_in,data_out)

data_in = train_X_nu1
data_out = train_Z_nu1
# fit
mlp_reg_nu1 = MLPRegressor(hidden_layer_sizes=(30,20),
                       max_iter = 1000,activation = 'relu',
                       solver = 'adam')

mlp_reg_nu1.fit(data_in,data_out)

data_in = train_X_nu2
data_out = train_Z_nu2
# fit
mlp_reg_nu2 = MLPRegressor(hidden_layer_sizes=(30,20),
                       max_iter = 1000,activation = 'relu',
                       solver = 'adam')

mlp_reg_nu2.fit(data_in,data_out)

# dataset de test

grid_test = init_grid([10,10,200,200,0,0])
X_test,Y_test = coordinate(grid_test)
X_test,Y_test = X_test.flatten(),Y_test.flatten()
predict_X = np.stack((X_test,Y_test),-1)
predict_X_uni = scaler_uni.transform(predict_X)
predict_X_nu1 = scaler_nu1.transform(predict_X)
predict_X_nu2 = scaler_nu2.transform(predict_X)


Z_exact = f(X_test,Y_test)
Z_predict_uni = np.abs(mlp_reg_uni.predict(predict_X_uni))
Z_predict_nu1 = np.abs(mlp_reg_nu1.predict(predict_X_nu1))
Z_predict_nu2 = np.abs(mlp_reg_nu2.predict(predict_X_nu2))


u = ((Z_exact - Z_predict_uni)** 2).sum()
v = ((Z_exact - Z_exact.mean()) ** 2).sum()
score_uniforme = int((1-(u/v))*100)

u = ((Z_exact - Z_predict_nu1)** 2).sum()
v = ((Z_exact - Z_exact.mean()) ** 2).sum()
score_raffine1 = int((1-(u/v))*100)

u = ((Z_exact - Z_predict_nu2)** 2).sum()
v = ((Z_exact - Z_exact.mean()) ** 2).sum()
score_raffine2 = int((1-(u/v))*100)


#%% résultats

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,12))
map1 = ax1.scatter(X_test,Y_test,c=Z_predict_uni,s=1,cmap='jet')
ax1.axis('square')
fig.colorbar(map1,ax=ax1,shrink=0.6)
ax1.set_title('prédiction à partir du dataset uniforme')

map2 = ax2.scatter(X_test,Y_test,c=Z_predict_nu1,s=1,cmap='jet')
ax2.axis('square')
fig.colorbar(map2,ax=ax2,shrink=0.6)
ax2.set_title('prédiction à partir du dataset raffiné sur l erreur')

map3 = ax3.scatter(X_test,Y_test,c=Z_predict_nu2,s=1,cmap='jet')
ax3.axis('square')
fig.colorbar(map3,ax=ax3,shrink=0.6)
ax3.set_title('prédiction à partir du dataset raffiné sur le gradient')

print('score uniforme :',score_uniforme,"%")
print('score raffine sur erreur :',score_raffine1,"%")
print('score raffine sur gradient :',score_raffine2,"%")


#%% statistiques

Score_u,Score_nu1,Score_nu2 = [],[],[]
for i in range(100):
    print(i)
    data_in = train_X_uni
    data_out = train_Z_uni
    # fit
    mlp_reg_uni = MLPRegressor(hidden_layer_sizes=(30,20),
                           max_iter = 1000,activation = 'relu',
                           solver = 'adam')

    mlp_reg_uni.fit(data_in,data_out)

    data_in = train_X_nu1
    data_out = train_Z_nu1
    # fit
    mlp_reg_nu1 = MLPRegressor(hidden_layer_sizes=(30,20),
                           max_iter = 1000,activation = 'relu',
                           solver = 'adam')

    mlp_reg_nu1.fit(data_in,data_out)

    data_in = train_X_nu2
    data_out = train_Z_nu2
    # fit
    mlp_reg_nu2 = MLPRegressor(hidden_layer_sizes=(30,20),
                           max_iter = 1000,activation = 'relu',
                           solver = 'adam')

    mlp_reg_nu2.fit(data_in,data_out)
    
    Z_predict_uni = np.abs(mlp_reg_uni.predict(predict_X_uni))
    Z_predict_nu1 = np.abs(mlp_reg_nu1.predict(predict_X_nu1))
    Z_predict_nu2 = np.abs(mlp_reg_nu2.predict(predict_X_nu2))


    u = ((Z_exact - Z_predict_uni)** 2).sum()
    v = ((Z_exact - Z_exact.mean()) ** 2).sum()
    score_uniforme = int((1-(u/v))*100)
    Score_u.append(score_uniforme)
    
    u = ((Z_exact - Z_predict_nu1)** 2).sum()
    v = ((Z_exact - Z_exact.mean()) ** 2).sum()
    score_raffine1 = int((1-(u/v))*100)
    Score_nu1.append(score_raffine1)

    u = ((Z_exact - Z_predict_nu2)** 2).sum()
    v = ((Z_exact - Z_exact.mean()) ** 2).sum()
    score_raffine2 = int((1-(u/v))*100)
    Score_nu2.append(score_raffine2)
    


Score_u,Score_nu1,Score_nu2 = np.array(Score_u),np.array(Score_nu1),np.array(Score_nu2)
#%% affichage

plt.plot(Score_u,'o',c='g',label='uniforme : %f pourcent'%(np.mean(Score_u)))
plt.plot(np.mean(Score_u)*np.ones(Score_u.size),'--',c='g')
plt.plot(Score_nu1,'o',c='b',label='raffinement sur erreur : %f pourcent'%np.mean(Score_nu1))
plt.plot(np.mean(Score_nu1)*np.ones(Score_u.size),'--',c='b')
plt.plot(Score_nu2,'o',c='r',label='raffinement sur gradient : %f pourcent'%np.mean(Score_nu2))
plt.plot(np.mean(Score_nu2)*np.ones(Score_u.size),'--',c='r')
plt.ylabel('Score de prédiction (%)')
plt.legend()

print('score uniforme :',np.mean(Score_u),"%")
print('score raffine sur erreur :',np.mean(Score_nu1),"%")
print('score raffine sur gradient :',np.mean(Score_nu2),"%")