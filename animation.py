# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:18:17 2022

@author: pchauris
"""

from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
camera = Camera(fig)
for i in range(10):
    plt.plot([i] * 10)
    camera.snap()
animation = camera.animate()
animation.save('celluloid_minimal.gif', writer = 'imagemagick')