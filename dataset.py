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

def load_PDNA543_HHM(x_hhm:dict, y_sites:dict, ws=11):
    x_pos, x_neg = [], []
    for key in x_hhm.keys():
        site = y_sites[key]
        hhm = x_hhm[key]
        head = np.zeros((30,))
        rear = np.zeros((1,30))
        for i in range(ws):
            hhm=np.insert(hhm,0,head,axis=0)
        for i in range(ws):
            hhm=np.append(hhm,rear,axis=0)
        n = len(site)
        for i in range(ws,ws+n):
            t = hhm[i-ws:i+ws+1]            
            if site[i-ws] == '1':
                x_pos.append(t)
            else:
                x_neg.append(t)
                
    return np.array(x_pos), np.array(x_neg)
    
if __name__ == "__main__":
    from prepareData import readPDNA543_hhm_sites
    (train_hhm, train_sites), (test_hhm, test_sites) = readPDNA543_hhm_sites()
    x_pos, x_neg = load_PDNA543_HHM(train_hhm, train_sites)
    
    