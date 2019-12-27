# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:03:35 2019

@author: lwzjc
"""
from __future__ import print_function
import keras
from keras import backend as K
import tensorflow as tf

from keras.layers import Input, Concatenate
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, CuDNNLSTM, Bidirectional, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
#from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# global variable
def semiSL2Dnet(shape,num_classes):
    l2value = 0.001
    x_input = Input(shape=shape,name="main_input")
    conv1 = Conv2D(32,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x_input) 
    
    conv2 = Conv2D(64,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(conv1)               
    
    conv3 = Conv2D(100,6,padding='same',activation="relu",kernel_regularizer=l2(l2value))(conv2) 
    
    pool1 = MaxPooling2D(2)(conv3)
    
    drop_1 = Dropout(0.3)
    x_a = drop_1(pool1)
    x_b = drop_1(pool1)
        
    flatten = Flatten()
    x_a = flatten(x_a)
    x_b = flatten(x_b)
        
    dense_1 = Dense(1024, activation='relu')
    x_a = dense_1(x_a)
    x_b = dense_1(x_b)
    
    drop_3 = Dropout(0.5,name="unsupLayer")
    x_a = drop_3(x_a)
    x_b = drop_3(x_b)
    
    out = Dense(num_classes, activation="softmax")(x_a)
    
    model = Model(inputs=x_input, outputs=[out, x_b])
    model.summary()
    
    return model
    
def semiSL2Dnet2(shape,num_classes):
    l2value = 0.001
    x_input = Input(shape=shape,name="main_input")
    conv1 = Conv2D(32,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x_input) 
    
    conv2 = Conv2D(64,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(conv1)               
      
    flatten = Flatten()(conv2)
            
    dense_1 = Dense(1024, activation='relu')(flatten)
        
    drop_1 = Dropout(0.5,name="unsupLayer")
    x_a = drop_1(dense_1)
    x_b = drop_1(dense_1)
    
    out = Dense(num_classes, activation="softmax")(x_a)
    
    model = Model(inputs=x_input, outputs=[out, x_b])
    model.summary()
    
    return model
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    