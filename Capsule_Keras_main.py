# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:37:11 2020

@author: falcon1
"""


from Capsule_Keras import Capsule

import numpy as np
import os
from Bio import SeqIO

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Reshape, Lambda
from tensorflow.keras import backend as K

from sklearn.metrics import accuracy_score, matthews_corrcoef,confusion_matrix
from sklearn.utils import class_weight, shuffle, resample
from sklearn.model_selection import train_test_split


def CapsNet(num_classes=2, dim_capsule=16, num_routing=3):
    input_image = Input(shape=(None,None,1))
    cnn = Conv2D(64, (3, 3), activation='relu')(input_image)
    cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
    cnn = AveragePooling2D((2,2))(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Reshape((-1, 128))(cnn)
    capsule = Capsule(num_classes, dim_capsule, num_routing, True)(cnn)
    output = Lambda(lambda x: K.sqrt(tf.keras.backend.sum(K.square(x), 2)), output_shape=(2,))(capsule)
    
    model = Model(inputs=input_image, outputs=output)
    model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.summary()
    return model

if __name__ == "__main__":
    # load PDNA543 testing data
    testdatafile = 'PDNA543TEST_HHM_11.npz'
    data = np.load(testdatafile, allow_pickle='True')
    x_test_pos, x_test_neg = data['pos'], data['neg']
    
    x_test = np.concatenate((x_test_pos, x_test_neg))
    y_test = [1] * x_test_pos.shape[0] + [0] * x_test_neg.shape[0]
    
    x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1).astype('float32')
    y_test = np.eye(2)[y_test].astype('float32')
    
    # load PDNA543 training data
    traindatafile = 'PDNA543_HHM_11.npz'
    data = np.load(traindatafile, allow_pickle='True')
    x_train_pos, x_train_neg = data['pos'], data['neg']
    
    num_pos, num_neg = x_train_pos.shape[0], x_train_neg.shape[0]
    R = int(np.floor( num_neg / num_pos))
    x_train_neg = shuffle(x_train_neg)
    
    y_pred = np.zeros(shape=(y_test.shape[0],))
    for i in range(R):
        if i < R-1:
            start_index, end_index = i * num_pos, (i+1) * num_pos
        if i == R-1:
            start_index, end_index = i * num_pos, num_neg
            
        x_train = np.concatenate((x_train_pos, x_train_neg[start_index:end_index]))
        y_train = [1] * num_pos + [0] * (end_index - start_index)
        y_train = np.eye(2)[y_train].astype('float32')
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.20, shuffle=True)
        x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1).astype('float32')
        x_val = x_val.reshape(-1, x_val.shape[1], x_val.shape[2], 1).astype('float32')
        
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        datagen.fit(x_train)
        
        model = CapsNet()
        
        model.fit(datagen.flow(x_train, y_train, batch_size=50),
          epochs=1,
          verbose=1,
          validation_data=(x_val, y_val))
        
        score = model.predict(x_test)
        y_pred = y_pred + np.argmax(score, axis=-1)
        K.clear_session()
    
    y_pred = y_pred / R
    y_p = (y_pred>0.5).astype(float)
    y_t = np.argmax(y_test, axis=-1)
    print('Test Accuracy:', accuracy_score(y_t, y_p))
    print('Test mattews-corrcoef', matthews_corrcoef(y_t, y_p))
    print('Test confusion-matrix', confusion_matrix(y_t, y_p))
            
    
    