# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 07:23:35 2020

@author: lwzjc
"""

import numpy as np
from SeqFormulate import greyPseAAC
from bls_enhence import broadnet_enhence
from sklearn import metrics

def load_data(file):
    data = np.load(file, allow_pickle=True)
    posseqs, negseqs = data['pos'], data['neg']
    weight = np.ones(24,) 
    weight[21], weight[23] = 0.01, 0.0001
    x_pos, x_neg = [],[]
    for seq in posseqs:    
        x_pos.append( greyPseAAC(seq, ['Hydrophobicity', 'ASE'], weight=weight, model=1, norm=False))
    for seq in negseqs:
        x_neg.append( greyPseAAC(seq, ['Hydrophobicity', 'ASE'], weight=weight, model=1))
    return np.array(x_pos), np.array(x_neg)

x_train_pos, x_train_neg = load_data('PDNA_543_train_7.npz')

x_train = np.concatenate((x_train_pos, x_train_neg))
y_train = [0 for _ in range(x_train_pos.shape[0])] + [1 for _ in range(x_train_neg.shape[0])]
y_train = np.array(y_train)

x_test_pos, x_test_neg = load_data('PDNA_543_test_7.npz')
x_test = np.concatenate((x_test_pos, x_test_neg))
y_test = [0 for _ in range(x_test_pos.shape[0])] + [1 for _ in range(x_test_neg.shape[0])]
y_test = np.array(y_test)

bls = broadnet_enhence(maptimes = 10, 
                           enhencetimes = 10,
                           traintimes = 10,
                           map_function = 'tanh',
                           enhence_function = 'sigmoid',
                           batchsize = 1, 
                           acc = 1,
                           step = 5,
                           reg = 0.001)
    
bls.fit(x_train,y_train)
predictlabel = bls.predict(x_test)

cm = metrics.confusion_matrix(y_test, predictlabel)
acc = metrics.accuracy_score(y_test, predictlabel)
mcc = metrics.matthews_corrcoef(y_test, predictlabel)

print("cm: ", cm)
print("accuracy: ", acc)
print("MCC: ", mcc)