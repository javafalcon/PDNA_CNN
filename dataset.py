#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:51:24 2020

@author: weizhong
"""
import numpy as np
from Formulater import Formulater
def load_PDNA543(trainNpzFile, testNpzFile):
    data = np.load(trainNpzFile)
    lsposseq, lsnegseq = data['pos'], data['neg']
    
    X_train_pos = []
    fmul = Formulater()
    for seq in lsposseq:
        #m1 = fmul.xiaoInfo(seq)/255
        #m2 = fmul.phychem(seq)
        m1 = fmul.OneHot(seq)/255
        m3 = fmul.accumulateFreq(seq)
        m3 = m3.reshape((len(m3),1))
        m = np.block([m1,m3])
        X_train_pos.append(m)
        
    X_train_neg = []
    for seq in lsnegseq:
        #m1 = fmul.xiaoInfo(seq)/255
        #m2 = fmul.phychem(seq)
        m1 = fmul.OneHot(seq)/255
        m3 = fmul.accumulateFreq(seq)
        m3 = m3.reshape((len(m3),1))
        m = np.block([m1,m3])
        X_train_neg.append(m) 
        
    data = np.load(testNpzFile)
    lsposseq, lsnegseq = data['pos'], data['neg'] 
    X_test_pos = []
    fmul = Formulater()
    for seq in lsposseq:
        #m1 = fmul.xiaoInfo(seq)/255
        #m2 = fmul.phychem(seq)
        m1 = fmul.OneHot(seq)/255
        m3 = fmul.accumulateFreq(seq)
        m3 = m3.reshape((len(m3),1))
        m = np.block([m1,m3])
        X_test_pos.append(m)
        
    X_test_neg = []
    for seq in lsnegseq:
        #m1 = fmul.xiaoInfo(seq)/255
        #m2 = fmul.phychem(seq)
        m1 = fmul.OneHot(seq)/255
        m3 = fmul.accumulateFreq(seq)
        m3 = m3.reshape((len(m3),1))
        m = np.block([m1,m3])
        X_test_neg.append(m) 
    
    return (np.array(X_train_pos), np.array(X_train_neg)), (np.array(X_test_pos), np.array(X_test_neg))