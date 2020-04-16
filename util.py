# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:29:54 2020

@author: lwzjc
"""
import numpy as np
def onehot(seq, row=60, col=20):
    if col == 20:
        amino_acids = "ACDEFHIGKLMNQPRSTVWY"
    elif col == 21:
        amino_acids = "ACDEFHIGKLMNQPRSTVWYX"
        
    x = np.zeros(shape=(row, col))
    n = len(seq) if len(seq)<=row else row
    for i in range(n):
        try:
            k = amino_acids.index(seq[i])
            x[i,k] = 1
        except:
            continue
    return x
