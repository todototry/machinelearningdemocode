# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 22:05:51 2013

@author: fandyst
"""
from mayavi import mlab  
import numpy as np
x, y, z, value = np.random.random((4, 40))
mlab.points3d(x, y, z, value)