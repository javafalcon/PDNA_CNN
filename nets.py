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
    x_input = Input(shape=shape,name="main_input")
    x1_1 = Conv2D(32,1,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x_input)
    x1_2 = Conv2D(32,1,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x_input)
    x1_2 = Conv2D(32,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x1_2)
    x1_3 = Conv2D(32,1,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x_input)
    x1_3 = Conv2D(32,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x1_3)
    x1_3 = Conv2D(32,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x1_3)
    x1_3 = Conv2D(32,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x1_3)   
    x = Concatenate()([x1_1,x1_2,x1_3])
        
    x = Flatten()(x)
            
    dense_1 = Dense(100, activation='relu')
    x_a = dense_1(x)
    x_b = dense_1(x)
    
    drop_3 = Dropout(0.3, name="unsupLayer")
    x_a = drop_3(x_a)
    x_b = drop_3(x_b)
    
    out = Dense(num_classes, activation="softmax")(x_a)
    
    model = Model(inputs=x_input, outputs=[out, x_b])
    model.summary()
    
    return model
    
def supLearnNet(shape,num_classes):
    l2value = 0.001
    x_input = Input(shape=shape,name="main_input")
    """
    x = Conv2D(32,(9,1),padding='same',activation="relu",kernel_regularizer=l2(l2value))(x_input) 
    x = MaxPooling2D(pool_size=(7,1))(x)
    x = Dropout(0.3)(x)
    """
    
    x_input = Input(shape=shape,name="main_input")
    x1_1 = Conv2D(32,1,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x_input)
    x1_2 = Conv2D(32,1,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x_input)
    x1_2 = Conv2D(32,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x1_2)
    x1_3 = Conv2D(32,1,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x_input)
    x1_3 = Conv2D(32,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x1_3)
    x1_3 = Conv2D(32,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x1_3)
    x1_3 = Conv2D(32,3,padding='same',activation="relu",kernel_regularizer=l2(l2value))(x1_3)   
    x = Concatenate()([x1_1,x1_2,x1_3])
    
    """
    x2_1 = Conv2D(100,1,padding='same',activation="relu")(x)
    x2_2 = Conv2D(100,1,padding='same',activation="relu")(x)
    x2_2 = Conv2D(100,3,padding='same',activation="relu")(x2_2)
    x2_3 = Conv2D(100,1,padding='same',activation="relu")(x)
    x2_3 = Conv2D(100,3,padding='same',activation="relu")(x2_3)
    x2_3 = Conv2D(100,3,padding='same',activation="relu")(x2_3)
    x2_3 = Conv2D(100,3,padding='same',activation="relu")(x2_3)
    x = Concatenate()([x2_1,x2_2,x2_3])
    """
    x = Flatten()(x)
    x = Dense(100,activation="relu",kernel_regularizer=l2(l2value))(x)
    x = Dropout(0.5)(x)
    out = Dense(2,activation="softmax")(x)
    
    model = Model(inputs=x_input, outputs=out)
    model.summary()
    
    return model
                    

                    
                    
                    
                    
                    