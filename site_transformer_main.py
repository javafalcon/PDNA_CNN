# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:39:12 2020

@author: lwzjc
"""
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import metrics
from sites_transformer import Encoder, create_padding_mask
from tools import plot_history

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

def transformer_train(x_train, y_train, x_test, y_test, n_layers,
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

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Train
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(x_train, y_train, batch_size=100, epochs=80, 
        validation_split=0.2)
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


seqlen = 15
vocab_size = 23
embed_dim = 16  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer
n_layers = 4
drop_rate = 0.2

x_pos, x_neg = load_seq_data('PDNA_224_7.npz')
y_true, y_pred = crosseval(x_pos, x_neg,5, n_layers,
          embed_dim, num_heads, ff_dim, seqlen, vocab_size,drop_rate)

cm = metrics.confusion_matrix(y_true, y_pred)
acc = metrics.accuracy_score(y_true, y_pred)
mcc = metrics.matthews_corrcoef(y_true, y_pred)

print("cm: ", cm)
print("accuracy: ", acc)
print("MCC: ", mcc)