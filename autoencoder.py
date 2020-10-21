#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:12:42 2020

@author: weizhong
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, Sequential
from prepareData import readPDNA543_seqs_sites, buildBenchmarkDataset2
from sklearn.utils import shuffle

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Sequential([
            layers.Input(shape=(31,24,1)),
            layers.Conv2D(16,(3,3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2)
            ])
        self.decoder = Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')
            ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def sample(vect, s=1.0):
    preds = np.asarray(vect).astype("float64")
    preds = np.log(preds + 0.000001) / s
    exp_preds = np.exp(preds)    
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def seqsEncode(seqs, winsize, chars, char_indices):
    x = np.zeros((len(seqs), winsize, len(chars)), dtype=np.bool)
    for i, seq in enumerate(seqs):
        for t, char in enumerate(seq):
            x[i, t, char_indices[char]] = 1
    return x

def Decode2Seqs(y_preds, indices_char):
    seqs = []
    for i in range(y_preds.shape[0]):
        y = y_preds[i].reshape((31, 24))
        seq = []
        for j in range(31):
            seq.append(indices_char[sample(y[j])])
        seqs.append("".join(seq))
    return seqs
            
amino_acids = 'ARNDCQEGHILKMFPSTWYV#'
chars = list(amino_acids)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

winsize, latent_dim = 31, 64
(train_seqs, train_sites), (test_seqs, test_sites) = readPDNA543_seqs_sites()
buildBenchmarkDataset2(train_seqs, train_sites, winsize//2, 'PDNA_543_train_15.npz')
buildBenchmarkDataset2(test_seqs, test_sites, winsize//2, 'PDNA_543_test_15.npz') 
train_data = np.load(allow_pickle=True, file='PDNA_543_train_15.npz')
train_pos, train_neg = train_data['pos'], train_data['neg']  
test_data = np.load(allow_pickle=True, file='PDNA_543_test_15.npz')
test_pos, test_neg = test_data['pos'], test_data['neg']

seqs_train = np.concatenate((train_pos, train_neg))
seqs_train = shuffle(seqs_train)
x = seqsEncode(seqs_train, winsize, chars, char_indices)
x = x.reshape((-1, 31, 24, 1))
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss', verbose=1, patience=5, restore_best_weights=True
    )
autoencoder = Autoencoder(latent_dim)    
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(x, x, epochs=200, batch_size=200, shuffle=True, 
                validation_split=0.2, callbacks=[early_stopping])

x_train_pos = seqsEncode(train_pos, winsize, chars, char_indices)
y_test = autoencoder.predict(x_train_pos)
t = Decode2Seqs(y_test[5:8], indices_char)



