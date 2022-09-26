# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:41:49 2022

@author: dngrn
"""

import ot
import numpy as np

rng = np.random.default_rng(12345)

#Generate Matrices
mat = rng.random((3, 3))
const = np.ones_like(mat)

#Normalize so it sums to 1
mat = mat / np.sum(mat)
const = const / np.sum(const)