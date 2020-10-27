#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:51:24 2020

@author: weizhong
"""
import numpy as np
from Formulater import Formulater

def readPDNA543_seqs_sites():
    from Bio import SeqIO
    train_seqs = {}
    train_sites = {}
    for seq_record in SeqIO.parse('PDNA_Data/TargetDNA/PDNA-543_sequence.fasta', 'fasta'):
        train_seqs[seq_record.id] = str(seq_record.seq)
    
    for site_record in SeqIO.parse('PDNA_Data/TargetDNA/PDNA-543_label.fasta', 'fasta'):
        train_sites[site_record.id] = str(site_record.seq)
    
    test_seqs, test_sites = {}, {}
    for seq_record in SeqIO.parse('PDNA_Data/TargetDNA/PDNA-TEST_sequence.fasta', 'fasta'):
        test_seqs[seq_record.id] = str(seq_record.seq)
    
    for site_record in SeqIO.parse('PDNA_Data/TargetDNA/PDNA-TEST_label.fasta', 'fasta'):
        test_sites[site_record.id] = str(site_record.seq)
    return (train_seqs, train_sites), (test_seqs, test_sites)


def readPDNA543_hhm_seqs_sites():    
    import os
    from HHSuite import read_hhm
    from Bio import SeqIO
    
    traindir = 'PDNA_Data/PDNA543_hhm'
    testdir = 'PDNA_Data/PDNA543TEST_hhm'
    
    x_train = {}
    train_sites = {}
    train_seqs = {}
    for file in os.listdir(traindir):
        r = read_hhm(os.path.join(traindir, file))
        x_train[file] = r[1]
        train_seqs[file] = "".join(r[0])
        
    for site_record in SeqIO.parse('PDNA_Data/TargetDNA/PDNA-543_label.fasta', 'fasta'):
        train_sites[site_record.id] = str(site_record.seq)
    
    x_test = {}
    test_sites = {}
    test_seqs = {}
    for file in os.listdir(testdir):
        r = read_hhm(os.path.join(testdir,file))
        x_test[file] = r[1]
        test_seqs[file] = "".join(r[0])
        
    for site_record in SeqIO.parse('PDNA_Data/TargetDNA/PDNA-TEST_label.fasta', 'fasta'):
        test_sites[site_record.id] = str(site_record.seq) 
        
    return (x_train, train_seqs, train_sites), (x_test, test_seqs, test_sites) 

def readPDNA543_hhm_sites():    
    import os
    from HHSuite import read_hhm
    from Bio import SeqIO
    
    traindir = 'PDNA_Data/PDNA543_hhm'
    testdir = 'PDNA_Data/PDNA543TEST_hhm'
    
    x_train = {}
    train_sites = {}
    for file in os.listdir(traindir):
        r = read_hhm(os.path.join(traindir, file))
        x_train[file] = r[1]
        
    for site_record in SeqIO.parse('PDNA_Data/TargetDNA/PDNA-543_label.fasta', 'fasta'):
        train_sites[site_record.id] = str(site_record.seq)
    
    x_test = {}
    test_sites = {}
    for file in os.listdir(testdir):
        r = read_hhm(os.path.join(testdir,file))
        x_test[file] = r[1]
        
    for site_record in SeqIO.parse('PDNA_Data/TargetDNA/PDNA-TEST_label.fasta', 'fasta'):
        test_sites[site_record.id] = str(site_record.seq) 
        
    return (x_train, train_sites), (x_test, test_sites) 

"""
params:
    @seqs sequence of proteins. For example: key1:'AVKKISQYACQRRTTLNNY'
    @sites the sites of protein-DNA. For example:key1:'0001100001001000000'
    @datefile the file save the data
"""    
def gen_PDNA543_accXiaoInfo(seqs:dict, sites:dict, datafile:str, ws=11):
    import os
    if os.path.exists(datafile):
        data = np.load(datafile, allow_pickle='True')
        return data['pos'], data['neg'] 
    else:
        fmul = Formulater()
        x_pos, x_neg = [],[]
        for key in seqs.keys():
            site = sites[key]
            seq = list(seqs[key])
            for i in range(ws):
                seq.insert(0,'X')
                seq.append('X')
            for i in range(ws, ws+len(site)):
                t = fmul.accumulateXiaoInfo(seq[i-ws:i+ws+1])
                if site[i-ws] == '1':
                    x_pos.append(t)
                else:
                    x_neg.append(t)
        np.savez(datafile, pos=x_pos, neg=x_neg)    
        return (np.array(x_pos), np.array(x_neg))
    

def gen_PDNA543_HHM(x_hhm:dict, y_sites:dict, datafile:str, ws=11):
    import os
    if os.path.exists(datafile):
        data = np.load(datafile, allow_pickle='True')
        return data['pos'], data['neg']
    else:
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
        np.savez(datafile, pos=x_pos, neg=x_neg)            
        return np.array(x_pos), np.array(x_neg)

def gen_PDNA543_HHM_accXiaoInfo(x_hhm:dict, seqs:dict, y_sites:dict, datafile:str, ws=11):
    import os
    if os.path.exists(datafile):
        data = np.load(datafile, allow_pickle='True')
        return data['pos'], data['neg']
    else:
        
        fmul = Formulater()
        x_pos, x_neg = [], []
        for key in x_hhm.keys():
            site = y_sites[key]
            hhm = x_hhm[key]
            seq = list(seqs[key])
            head = np.zeros((30,))
            rear = np.zeros((1,30))
            for i in range(ws):
                hhm=np.insert(hhm,0,head,axis=0)
                seq.insert(0,'X')
            for i in range(ws):
                hhm=np.append(hhm,rear,axis=0)
                seq.append('X')
            n = len(site)
            for i in range(ws,ws+n):
                t1 = hhm[i-ws:i+ws+1]  
                t2 = fmul.accumulateXiaoInfo(seq[i-ws:i+ws+1])
                t = np.block([t1,t2])
                if site[i-ws] == '1':
                    x_pos.append(t)
                else:
                    x_neg.append(t)
        np.savez(datafile, pos=x_pos, neg=x_neg)            
        return np.array(x_pos), np.array(x_neg)
    
if __name__ == "__main__":
    #from prepareData import readPDNA543_hhm_sites
    (train_hhm, train_seqs, train_sites), (test_hhm, test_seqs, test_sites) = readPDNA543_hhm_seqs_sites()
    #x_pos, x_neg = gen_PDNA543_HHM_accXiaoInfo(train_hhm,train_seqs,train_sites,'PDNA543_hhm_accxiaoinfo_11.npz',11)
    #x_pos, x_neg = gen_PDNA543_HHM_accXiaoInfo(test_hhm,test_seqs,test_sites,'PDNA543TEST_hhm_accxiaoinfo_11.npz',11)
    train_pos, train_neg = gen_PDNA543_HHM(train_hhm, train_sites,'PDNA543_HHM_15.npz',15)
    test_pos, test_neg = gen_PDNA543_HHM(test_hhm, test_sites, 'PDNA543TEST_HHM_15.npz',15)
    #from prepareData import slipwindown_v1
    #(train_seqs, train_sites), (test_seqs, test_sites) = readPDNA543_seqs_sites()
    #slipwindown_v1(train_seqs, train_sites, 15, 'PDNA543_seqs_15.npz')
    #slipwindown_v1(test_seqs, test_sites, 15, 'PDNA543TEST_seqs_15.npz')
    
    