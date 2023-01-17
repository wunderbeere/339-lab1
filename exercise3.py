# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:40:00 2023

@author: Bhuvan
"""

import numpy as np
import matplotlib.pyplot as plt
import random as r

Nsamp = 10

gauss_values = np.zeros(int(Nsamp))
meangauss = np.zeros(int(Nsamp))
trials = range(int(Nsamp))

for i in trials:
    gauss_values[i] = r.gauss(0.0564405, 1.0561) 
    for j in trials:
        meangauss[j] = np.mean(gauss_values)

     
