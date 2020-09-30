                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               # -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:39:12 2020

@author: lwzjc
"""
import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import metrics
from sites_transformer import Encoder, create_padding_mask
from tools import plot_history
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from collections import Counter

# Define our custom loss function
def focal_loss(y_true, y_pred):
    gamma, alpha= 2.0, 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def seqs_to_vec(seqs):
    amino_acids = "#ARNDCQEGHILKMFPSTWYVX"
    regexp = re.compile('[^ARNDCQEGHILKMFPSTWYVX#]')
    x = []
    for seq in seqs:
        seq = regexp.sub('X', seq)
        t = []
        for a in seq:
            t.append(amino_acids.index(a))
        x.append(t) 
    return np.array(x)

def load_seq_data(file):
    data = np.load(file, allow_pickle=True)
    posseqs, negseqs = data['pos'], data['neg']
    x_pos = seqs_to_vec(posseqs)
    x_neg = seqs_to_vec(negseqs)
    return x_pos, x_neg

def transformer_train(x_train, y_train, x_weight, x_test, y_test, n_layers,
                      embed_dim, num_heads, ff_dim, seqlen, vocab_size,drop_rate):
    inputs = layers.Input(shape=(seqlen, ))
    encode_padding_mask = create_padding_mask(inputs)
    encoder = Encoder(n_layers=n_layers, d_model=embed_dim, n_heads=num_heads, ffd=ff_dim,
             seq_len=seqlen, input_vocab_size=vocab_size, dropout_rate=drop_rate)
    x = encoder(inputs, False, encode_padding_mask)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    # Train
    # method 1: weight balancing
    #class_weight = {0:1, 1:1}
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    
    # method 2: Focal loss
    #model.compile(loss=[focal_loss], metrics=["accuracy"], optimizer="adam") 
    
    model.summary()

    history = model.fit(x_train, y_train, batch_size=100, epochs=50, sample_weight=x_weight,
        validation_split=0.1)
    plot_history(history)
    
    prediction = model.predict(x_test)
    y_pred = np.argmax(prediction, axis=1)
    
    return y_pred

def crosseval(x_pos, x_neg, k, n_layers,
             embed_dim, num_heads, ff_dim, 
             seqlen, vocab_size,drop_rate):
    kf = KFold(n_splits=k, shuffle=True, random_state=43)
    kf_pos_train, kf_pos_test = [],[]
    for train_indx, test_indx in kf.split(x_pos):
        train, test = [], []
        for i in train_indx:
            train.append(x_pos[i])
        for i in test_indx:
            test.append(x_pos[i])
            
        kf_pos_train.append(np.array(train))
        kf_pos_test.append(np.array(test))
    
    kf_neg_train, kf_neg_test = [],[]
    for train_indx, test_indx in kf.split(x_neg):
        train, test = [], []
        for i in train_indx:
            train.append(x_neg[i])
        for i in test_indx:
            test.append(x_neg[i])
            
        kf_neg_train.append(np.array(train))
        kf_neg_test.append(np.array(test))
    
    y_t, y_p = np.zeros(0,), np.zeros(0,)
    for i in range(k):
        pos_train, neg_train = kf_pos_train[i], kf_neg_train[i]
        pos_test, neg_test = kf_pos_test[i], kf_neg_test[i]
        
        pos_reps = np.tile(pos_train, (14,1))
        x_train = np.concatenate((pos_reps, neg_train))
        y_train = [1 for _ in range(pos_reps.shape[0])] + [0 for _ in range(neg_train.shape[0])]
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
        x_train, y_train = shuffle(x_train, y_train)
        
        x_test = np.concatenate((pos_test, neg_test))
        y_test = [1 for _ in range(pos_test.shape[0])] + [0 for _ in range(neg_test.shape[0])]
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
        
        tf.keras.backend.clear_session()
        pred = transformer_train(x_train, y_train, x_test, y_test, n_layers,
                      embed_dim, num_heads, ff_dim, seqlen, vocab_size,drop_rate)
        
        y_t = np.concatenate((y_t, np.argmax(y_test, axis=1)))
        y_p = np.concatenate((y_p, pred))
        
    return y_t, y_p    


seqlen = 31
vocab_size = 22
embed_dim = 18  # Embedding size for each token
num_heads = 6  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer
n_layers = 8
drop_rate = 0.2

"""
# dataset: pdna-224 
x_pos, x_neg = load_seq_data('PDNA_224_7.npz')
y_true, y_pred = crosseval(x_pos, x_neg,5, n_layers,
          embed_dim, num_heads, ff_dim, seqlen, vocab_size,drop_rate)

"""
# dataset: pdna-543
x_train_pos, x_train_neg = load_seq_data('PDNA_543_train_15.npz')
#x_train_pos = np.tile(x_train_pos, (14,1))

x_train = np.concatenate((x_train_pos, x_train_neg))
y_train = [0 for _ in range(x_train_pos.shape[0])] + [1 for _ in range(x_train_neg.shape[0])]
x_weight = [20 for _ in range(x_train_pos.shape[0])] + [1 for _ in range(x_train_neg.shape[0])]
y_train = keras.utils.to_categorical(y_train, num_classes=2)
x_weight = np.array(x_weight)
#print('Original dataset shape %s' % Counter(y_train))

# testing data
x_test_pos, x_test_neg = load_seq_data('PDNA_543_test_15.npz')
x_test = np.concatenate((x_test_pos, x_test_neg))
y_test = [0 for _ in range(x_test_pos.shape[0])] + [1 for _ in range(x_test_neg.shape[0])]
y_test = keras.utils.to_categorical(y_test, num_classes=2)

# ensembling with under-sampling majority class
y_score = np.zeros(shape=(y_test.shape[0],))
y_true = np.argmax(y_test, axis=1)

"""
us = NearMiss()
x_train_res, y_train_res = us.fit_resample(x_train, y_train) 
print('Near Miss Under resampled dataset shape %s' % Counter(y_train_res))

sm = SMOTE(random_state=42)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train) 
print('SMOTE over-resampled dataset shape %s' % Counter(y_train_res))
"""

x_train, y_train, x_weight = shuffle(x_train, y_train, x_weight)
K.clear_session()
y_pred = transformer_train(x_train, y_train, x_weight, x_test, y_test, n_layers,
                  embed_dim, num_heads, ff_dim, seqlen, vocab_size, drop_rate)


# predict performance
cm = metrics.confusion_matrix(y_true, y_pred)
acc = metrics.accuracy_score(y_true, y_pred)
mcc = metrics.matthews_corrcoef(y_true, y_pred)

print("cm: ", cm)
print("accuracy: ", acc)
print("MCC: ", mcc)