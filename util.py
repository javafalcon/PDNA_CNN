# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:29:54 2020

@author: lwzjc
"""
import numpy as np
from aaindexValues import aaindex1PCAValues
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

def pcaval(seqs:list):
    aadict = aaindex1PCAValues()
    x = []
    for i in range(len(seqs)):
        seq = seqs[i]
        n = len(seq)
        t = np.zeros(shape=(n,16))
        for j in range(n):
            if seq[j] in aadict.keys():
                t[j,:15] = aadict[seq[j]]
                t[j,15] = 0
            else:
                t[j] = np.zeros(16,)
                if seq[j] == '#':
                    t[j,15] = 1
        x.append(t)
    return np.array(x)

if __name__ == "__main__":
    seqs=['MKRRIRRERNKMAAAKSRNRRRELTDTLQAETDQLEDEKSALQTEIANLLKEKEKL',
          'MALTNAQILAVIDSWEETVGQFPVITHHVPLGGGLQGTLHCYEIPLAAPYGVGFAKNGPTRWQYKRTINQVV']
    x = pcaval(seqs)