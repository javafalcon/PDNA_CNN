#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:54:46 2020

@author: weizhong
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, Sequential
from prepareData import readPDNA543_seqs_sites, buildBenchmarkDataset2
from sklearn.utils import shuffle

# laod sequential data......
winsize, latent_dim = 31, 64
(train_seqs, train_sites), (test_seqs, test_sites) = readPDNA543_seqs_sites()
buildBenchmarkDataset2(train_seqs, train_sites, winsize//2, 'PDNA_543_train_15.npz')
buildBenchmarkDataset2(test_seqs, test_sites, winsize//2, 'PDNA_543_test_15.npz') 
train_data = np.load(allow_pickle=True, file='PDNA_543_train_15.npz')
train_pos, train_neg = train_data['pos'], train_data['neg']  
test_data = np.load(allow_pickle=True, file='PDNA_543_test_15.npz')
test_pos, test_neg = test_data['pos'], test_data['neg']

amino_acids = '#ARNDCQEGHILKMFPSTWYV'
chars = list(amino_acids)
char_indices = dict((c, i) for i, c in enumerate(chars))

def onehotSeqs(seqs):
    x = np.zeros((len(seqs), winsize, len(chars)), dtype=np.float)
    for i, seq in enumerate(seqs):
        for t, char in enumerate(seq):
            x[i, t, char_indices[char]] = 1
    return x

# sequences encode......
x_pos_train = onehotSeqs(train_pos)
y_pos_train = [0] * len(x_pos_train)
y_neg_train = [1] * len(x_neg_train)
y_pos_train = [0] * len(train_pos)
y_neg_train = [1] * len(train_neg)
y_train = [0] * len(train_pos) + [1] * len(train_neg)
neg, pos = np.bincount(y_train)
y_train = [1] * len(train_pos) + [0] * len(train_neg)
neg, pos = np.bincount(y_train)
initial_bias = np.log([pos/neg])


METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    ]
total = neg + pos
weight_for_0 = (1 / neg) * (total) / 2.0
weight_for_1 = (1 / pos) * (total) / 2.0