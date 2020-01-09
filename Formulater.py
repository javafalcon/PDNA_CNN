#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:55:29 2020

@author: weizhong
"""
import numpy as np
from  AminoAcidCode import *
class Formulater:
    def __init__(self, num_aa=20):
        self.num_aa = num_aa
        if num_aa == 21:
            self.amino_acid_alp = "ACDEFHIGKLMNQPRSTVWYX"
        elif num_aa == 20:
            self.amino_acid_alp = "ACDEFHIGKLMNQPRSTVWY"
            
    def OneHot(self, seq):
        fmul = []
        for s in seq:
            x = np.zeros((len(self.amino_acid_alp),))
            try:
                k = self.amino_acid_alp.index(s)
                x[k] = 1
                fmul.append(x)
            except ValueError:
                fmul.append(x)
                continue
                    
        return np.array(fmul, dtype=float)   

    
    def xiaoInfo(self, seq):
        x = np.zeros(shape=(len(seq), 5))
        for i in range(len(seq)):
            x[i] = np.zeros(shape=(5,))
            if seq[i] in code['XiaoInfoCode'].keys():
                c = code['XiaoInfoCode'][seq[i]]
                for k in range(5):
                    x[i,k] = int(c[k])
        return x    
    
    def phychem(self, seq):
        x = np.zeros(shape=(len(seq), 7))
        for i in range(len(seq)):
            x[i] = np.zeros(shape=(7,))
            if seq[i] in code['phychemCodeLog'].keys():
                x[i] = code['phychemCodeLog'][seq[i]]
        return x
    
    def accumulateFreq(self, seq):
        x = np.zeros((len(seq),))
        aadict = dict()
        
        for i in range(len(seq)):
            k = aadict.get(seq[i],0)+1
            aadict[seq[i]] = k
            x[i] = k/(i+1)
        return x