#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:12:17 2020

@author: weizhong
"""
import numpy as np
# 读入PDNA-224数据。返回蛋白质序列及其dna结合位点在序列中的序号（从1开始计数）
# 返回两个字典。
# pdna_seqs_224  序列字典, key：蛋白质id  value: 蛋白质序列
# pdna_sites_224 结合位点字典, key：蛋白质id  value:位点序号组成的列表
def readPDNA224():
    pdna_seqs_224 = {}
    pdna_sites_224 = {}
    with open(r'PDNA_Data/Supplementary_data_S1_010913.doc','r',encoding='ISO-8859-1') as fr:
        sequencesFlag, sitesFlag = False, False
        for line in fr:
            line = line.replace("\n","")
            line = line.replace("\x0c","")
            if 'The sequences of PDNA-224' in line:
                sequencesFlag = True
                continue
            elif 'The binding sites information' in line:
                sequencesFlag, sitesFlag =False, True
                continue
            if sequencesFlag:
                if len(line.strip()) == 0:
                    continue
                if line.startswith('>'):
                    key = line[1:]
                else:
                    pdna_seqs_224[key] = pdna_seqs_224.get(key,'') + line
                
            elif sitesFlag:
                if len(line.strip()) == 0:
                    continue
                if line.startswith('>'):
                    key = line[1:]
                    pdna_sites_224[key] = []
                else:
                    line = line.rstrip("\t")
                    for s in line.split("\t"):
                        try:
                            pdna_sites_224[key].append(int(s))
                        except ValueError:
                            break
    return pdna_seqs_224, pdna_sites_224

def readPDNA224_hhm_seqs_sites():    
    import os
    from HHSuite import read_hhm
    
    traindir = 'PDNA_Data/PDNA224_hhm'
    
    seqs, sites = readPDNA224()
    
    x_train = {}
    train_seqs = {}
    for file in os.listdir(traindir):
        r = read_hhm(os.path.join(traindir, file))
        x_train[file] = r[1]
        train_seqs[file] = "".join(r[0])
                
    return (x_train, train_seqs, sites)

def gen_PDNA224_HHM(x_hhm:dict, y_sites:dict, datafile:str, ws=11):
    import os
    if os.path.exists(datafile):
        data = np.load(datafile, allow_pickle='True')
        return data['pos'], data['neg']
    else:
        x_pos, x_neg = [], []
        for key in x_hhm.keys():
            site = y_sites[key]
            hhm = x_hhm[key]
            n = len(hhm)
            head = np.zeros((30,))
            rear = np.zeros((1,30))
            for i in range(ws):
                hhm=np.insert(hhm,0,head,axis=0)
            for i in range(ws):
                hhm=np.append(hhm,rear,axis=0)
            
            for i in range(ws,ws+n):
                t = hhm[i-ws:i+ws+1]            
                if i-ws+1 in site:
                    x_pos.append(t)
                else:
                    x_neg.append(t)
        np.savez(datafile, pos=x_pos, neg=x_neg)            
        return np.array(x_pos), np.array(x_neg)

def load_kf_data(benckmarkFile='PDNA224_hhm_11.npz', k=5):
    from sklearn.model_selection import KFold
    data = np.load(benckmarkFile, allow_pickle='True')
    x_pos, x_neg = data['pos'], data['neg']
    
    kf = KFold(n_splits=k)
    kf_x_pos_train, kf_x_pos_test = [],[]
    for train_index, test_index in kf.split(x_pos):
        kf_x_pos_train.append(x_pos[train_index])
        kf_x_pos_test.append(x_pos[test_index])
    
    kf_x_neg_train, kf_x_neg_test = [],[]
    for train_index, test_index in kf.split(x_neg):
        kf_x_neg_train.append(x_neg[train_index])
        kf_x_neg_test.append(x_neg[test_index])
    
    return (kf_x_pos_train, kf_x_neg_train), (kf_x_pos_test, kf_x_neg_test)  
    
if __name__ == "__main__":
    (hhms,seqs,sites) = readPDNA224_hhm_seqs_sites()
    x_pos, x_neg = gen_PDNA224_HHM(hhms,sites,'PDNA224_HHM_15.npz',15)