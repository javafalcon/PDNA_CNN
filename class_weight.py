#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:54:46 2020

@author: weizhong
"""
import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, Sequential
from prepareData import readPDNA543_seqs_sites, buildBenchmarkDataset2
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

def binencode(n, m=5):
    t = np.zeros((m,), dtype=float)
    j = m-1
    while n > 0:
        if n%2 == 1:
            t[j] = 1.0

        n = n//2
        j -= 1
    return t

# amino_acids onehot + position binary encode
def onehotSeqs(seqs):
    x = np.zeros((len(seqs), winsize, len(chars)+5), dtype=np.float)
    for i, seq in enumerate(seqs):
        for t, char in enumerate(seq):
            x[i, t, char_indices[char]] = 1
            b = binencode(t, 5)
            x[i, t, len(chars):] = b
    return x

# sequences encode......
x_pos_train = onehotSeqs(train_pos)
x_neg_train = onehotSeqs(train_neg)
y_pos_train = np.array([1] * len(train_pos))
y_neg_train = np.array([0] * len(train_neg))

x_pos_test = onehotSeqs(test_pos)
x_neg_test = onehotSeqs(test_neg)
x_test = np.concatenate((x_pos_test, x_neg_test))
y_test = [1] * len(test_pos) + [0] * len(test_neg)
y_test = np.array(y_test)

# define metrics and model
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

def make_model(metrics = METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        
    regular = tf.keras.regularizers.l1(0.01)
    x = layers.Input(shape=(31,26,1))
    conv1 = layers.Conv2D(32, (5,5), strides=1,
                          padding='same', activation='relu', 
                          kernel_regularizer=regular,
                          name='conv1')(x)
    conv2 = layers.Conv2D(32, (5,5), padding='same', activation='relu', name='conv2')(conv1)
    pool1 = layers.MaxPool2D(pool_size=(2,2))(conv2)
    drop1 = layers.Dropout(0.25)(pool1)
    
    conv3 = layers.Conv2D(64,(5,5), padding='same',
                          activation='relu', name='conv3')(drop1)
    conv4 = layers.Conv2D(128, (5,5), activation='relu', name='conv4')(conv3)
    pool2 = layers.MaxPool2D()(conv4)
    drop2 = layers.Dropout(0.25)(pool2)
    
    flat = layers.Flatten()(drop2)
    dens1 = layers.Dense(512, activation='relu')(flat)
    drop3 = layers.Dropout(0.5)(dens1)
    out = layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)(drop3)
    
    model = Model(inputs=x, outputs=out)
    model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

    return model

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))
  
EPOCHS = 100
BATCH_SIZE = 100

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

weighted_model = make_model()
initial_weights = os.path.join('save_models','initial_weights')
weighted_model.save_weights(initial_weights)
weighted_model.load_weights(initial_weights)

"""
weight_for_0 = (1 / neg) * (total) / 2.0
weight_for_1 = (1 / pos) * (total) / 2.0
class_weight = {0: weight_for_0, 1: weight_for_1}
"""
ids = np.arange(len(train_pos))
choices = np.random.choice(ids, len(train_neg))

res_x_pos_train = x_pos_train[choices]
res_y_pos_train = y_pos_train[choices]

resampled_features = np.concatenate([res_x_pos_train, x_neg_train], axis=0)
resampled_labels = np.concatenate([res_y_pos_train, y_neg_train], axis=0)

order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

history = weighted_model.fit(
    resampled_features,
    resampled_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(x_test, y_test))

#train_predictions_weighted = weighted_model.predict(x_train, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(x_test, batch_size=BATCH_SIZE)
weighted_results = weighted_model.evaluate(x_test, y_test,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
  print(name, ': ', value)
print()

plot_cm(y_test, test_predictions_weighted)