# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:28:55 2019

@author: Administrator
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import os
import numpy as np
from sklearn.model_selection import KFold
from prepareData import readPDNA224, getTrainingDataset, genEnlargedData
def net(X_train, X_test, y_train, save_dir, model_name, batch_size, epochs):
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same', input_shape=(32,32,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # initiate RMSprop optimizer
    #opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer="Adam",
                  metrics=['accuracy'])
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    print('Saved trained model at %s ' % model_path)
    model.save(model_path)
    return model.predict(X_test)


batch_size = 32
num_classes = 2
epochs = 100
save_dir = os.path.join(os.getcwd(), 'save_models')
pseqs, psites = readPDNA224()
posseqs, negseqs = getTrainingDataset(pseqs,psites,11)

X_train_pos_ls, X_test_pos_ls = [], []
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(posseqs):
    X_train_pos, X_test_pos = [], []
    for k in train_index:
        X_train_pos.append(posseqs[k])
    for j in test_index:
        X_test_pos.append(posseqs[j])
    
    X_train_pos_ls.append(X_train_pos)
    X_test_pos_ls.append(X_test_pos)

X_train_neg_ls, X_test_neg_ls = [], []
for train_index, test_index in kf.split(negseqs):
    X_train_neg, X_test_neg = [], []
    for k in train_index:
        X_train_neg.append(negseqs[k])
    for j in test_index:
        X_test_neg.append(negseqs[j])
        
    X_train_neg_ls.append(X_train_neg)
    X_test_neg_ls.append(X_test_neg)

y_pred, y_targ = [], []
for i in range(5):
    X_train, y_train = genEnlargedData(X_train_pos_ls[i], X_train_neg_ls[i])
    
    X_test, y_test = genEnlargedData(X_test_pos_ls[i], X_test_neg_ls[i], 1)
       
    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    model_name = 'keras_pdna224_trained_5fold_model_{}.h5'.format(i)
    X_train = X_train.reshape(X_train.shape[0],32,32,1)
    X_test = X_test.reshape(X_test.shape[0],32,32,1)
    pred = net(X_train, X_test, y_train, save_dir, model_name, batch_size, epochs)
    y_pred += pred
    y_targ += y_test