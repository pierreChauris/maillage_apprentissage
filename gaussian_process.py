# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:34:42 2022

@author: pchauris
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=8, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on noise-free dataset")

#%%

from raffinement import *

grid = init_grid([8*np.pi,2,100,100,0,-1])
x,y = coordinate(grid)
X = np.stack((x,y),-1)
z = np.exp(-5*y**2)*x*np.sin(x)


plt.figure()
plt.scatter(x,y,c=z,cmap='jet')
plt.colorbar()

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(z.size), size=100, replace=False)
X_train, z_train = X[training_indices], z[training_indices]

noise_std = 0.1
z_train_noisy = z_train + rng.normal(loc=0.0, scale=noise_std, size=z_train.shape)

plt.figure()
plt.scatter(X_train[:,0],X_train[:,1],c=z_train_noisy,cmap='jet',vmin=np.min(z),vmax=np.max(z))
plt.colorbar()

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, z_train_noisy)
gaussian_process.kernel_

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

erreur = np.abs(z-mean_prediction)


plt.figure()
plt.scatter(x,y,c=erreur,cmap='jet')
plt.colorbar()