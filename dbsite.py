# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:42:23 2020

@author: lwzjc
"""
from aaindexValues import aaindex1Values
import numpy as np

AminoAcids = 'ARNDCQEGHILKMFPSTWYV'
aavals = aaindex1Values()
N = aavals.shape[0]

def formulateProt(seq):
    a = np.zeros(shape=(len(seq), N))
    for i in range(len(seq)):
        try:
            k = AminoAcids.index(seq[i])
        except ValueError: #if seq[i] is not 20 amino acids，            
            for j in range(N):
                a[i][j] = np.mean(aavals[j])#then use the mean value
        else:
            for j in range(N):          
                a[i][j] = aavals[j][k]
    return a

def load_data():
    train_data = np.load('PDNA_data\\PDNA_543_train_10.npz')
    test_data = np.load('PDNA_data\\PDNA_543_test_10.npz')
    train_pos_seq, train_neg_seq = train_data['pos'], train_data['neg']
    test_pos_seq, test_neg_seq = test_data['pos'], test_data['neg']
   
    train_X = []
    for seq in train_pos_seq:
        train_X.append(formulateProt(seq))
        
    for seq in train_neg_seq:
        train_X.append(formulateProt(seq))
    
    test_X = []
    for seq in test_pos_seq:
        test_X.append(formulateProt(seq))
        
    for seq in test_neg_seq:
        test_X.append(formulateProt(seq))

    train_y = np.zeros((len(train_X,)))
    train_y[:len(train_pos_seq)] = 1
    
    test_y = np.zeros((len(test_X,)))
    test_y[:len(test_pos_seq)] = 1
    
    return (train_X, train_y), (test_X, test_y)

def load_Onehot_data():
    from util import onehot
    train_data = np.load('PDNA_data\\PDNA_543_train_10.npz')
    test_data = np.load('PDNA_data\\PDNA_543_test_10.npz')
    train_pos_seq, train_neg_seq = train_data['pos'], train_data['neg']
    test_pos_seq, test_neg_seq = test_data['pos'], test_data['neg']
    
    train_X = []
    for seq in train_pos_seq:
        train_X.append(onehot(seq,21,20))
    
    for seq in train_neg_seq:
        train_X.append(onehot(seq,21,20))
    
    test_X = []
    for seq in test_pos_seq:
        test_X.append(onehot(seq,21,20))
    
    for seq in test_neg_seq:
        test_X.append(onehot(seq,21,20))
    
    train_y = np.zeros((len(train_X,)))
    train_y[:len(train_pos_seq)] = 1
    
    test_y = np.zeros((len(test_X,)))
    test_y[:len(test_pos_seq)] = 1

    return (train_X, train_y), (test_X, test_y)

if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = load_Onehot_data()
    np.savez('PDNA_543_onehot_10.npz', train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)
       

    