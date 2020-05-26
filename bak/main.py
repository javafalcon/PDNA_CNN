# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:28:55 2019

@author: Administrator
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
import os
from numba import cuda
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef
from prepareData import readPDNA224, getTrainingDataset, genEnlargedData,protsFormulateByChaosCode

# global variable
ws = 11
batch_size = 32
num_classes = 2
epochs = 30
save_dir = os.path.join(os.getcwd(), 'save_models')

def net(X_train, X_test, y_train, save_dir, model_name, batch_size, epochs):
    l2value = 0.001
    x_input = Input(shape=(6*ws+3,2,),name="main_input")
    x_1 = Conv1D(32,3,activation='relu',padding="same")(x_input)
    x_2 = Conv1D(32,5,activation='relu',padding="same")(x_input)
    x_3 = Conv1D(32,9,activation='relu',padding="same")(x_input)
    x_4 = Conv1D(32,15,activation='relu',padding="same")(x_input)
    #x_5 = Conv1D(32,15,activation='relu',padding="same",kernel_regularizer=l2(l2value))(x_input)
    
    x = Concatenate()([x_1, x_2, x_3, x_4])
    """
    x = Conv1D(64,5,activation='relu', padding="same",kernel_regularizer=l2(l2value))(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    """
    x = Conv1D(128,5,activation='relu', padding="same",kernel_regularizer=l2(l2value))(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(CuDNNLSTM(10))(x)
    x = Dropout(0.3)(x)
    
    out = Dense(1024, activation="relu")(x)
    out = Dropout(0.5)(out)
    
    out = Dense(2, activation="softmax")(out)
    
    model = Model(inputs=x_input, outputs=out)
    model.summary()

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              #callbacks = [EarlyStopping('val_loss',patience=5)],
              validation_split=0.2,
              shuffle=True)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    print('Saved trained model at %s ' % model_path)
    model.save(model_path)
    return model.predict(X_test)

def net2(X_train, X_test, y_train, save_dir, model_name, batch_size, epochs):
    
    x_input = Input(shape=(2*ws+1,21,1,),name="main_input")
    x1_1 = Conv2D(32,1,padding='same',activation="relu")(x_input)
    x1_2 = Conv2D(32,1,padding='same',activation="relu")(x_input)
    x1_2 = Conv2D(32,3,padding='same',activation="relu")(x1_2)
    x1_3 = Conv2D(32,1,padding='same',activation="relu")(x_input)
    x1_3 = Conv2D(32,3,padding='same',activation="relu")(x1_3)
    x1_3 = Conv2D(32,3,padding='same',activation="relu")(x1_3)
    x1_3 = Conv2D(32,3,padding='same',activation="relu")(x1_3)   
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
    """
    x3_1 = Conv2D(100,1,padding='same',activation="relu")(x)
    x3_2 = Conv2D(100,1,padding='same',activation="relu")(x)
    x3_2 = Conv2D(100,3,padding='same',activation="relu")(x3_2)
    x3_3 = Conv2D(100,1,padding='same',activation="relu")(x)
    x3_3 = Conv2D(100,3,padding='same',activation="relu")(x3_3)
    x3_3 = Conv2D(100,3,padding='same',activation="relu")(x3_3)
    x3_3 = Conv2D(100,3,padding='same',activation="relu")(x3_3)
    x = Concatenate()([x3_1,x3_2,x3_3])
    """
    x = Flatten()(x)
    #x = Dense(512,activation="relu")(x)
    #x = Dropout(0.5)(x)
    x = Dense(1024,activation="relu")(x)
    x = Dropout(0.5)(x)
    
    out = Dense(2,activation="softmax")(x)
    
    model = Model(inputs=x_input, outputs=out)
    model.summary()

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              shuffle=True)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    print('Saved trained model at %s ' % model_path)
    model.save(model_path)
    
    pred =  model.predict(X_test)
    K.clear_session()
    tf.reset_default_graph()
    
    return pred

def splitDataKF(Kf=5):
    pseqs, psites = readPDNA224()
    posseqs, negseqs = getTrainingDataset(pseqs,psites,ws)
    
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
    
    return (X_train_pos_ls, X_test_pos_ls), (X_train_neg_ls, X_test_neg_ls)

def predictor1():
    (X_train_pos_ls, X_test_pos_ls), (X_train_neg_ls, X_test_neg_ls) = splitDataKF()
    y_pred, y_targ = np.zeros((0,2)), np.zeros((0,2))
    for i in range(5):
        X_train, y_train = genEnlargedData(X_train_pos_ls[i], X_train_neg_ls[i])
        
        X_test, y_test = genEnlargedData(X_test_pos_ls[i], X_test_neg_ls[i], 1)
        print("{} fold training and test........".format(i))   
        print('x_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
        
        model_name = 'keras_pdna224_trained_5fold_model_{}.h5'.format(i)
        #X_train = X_train.reshape(X_train.shape[0],32,32,1)
        #X_test = X_test.reshape(X_test.shape[0],32,32,1)
        pred = net(X_train, X_test, y_train, save_dir, model_name, batch_size, epochs)
        
        y_t = np.argmax(y_test,-1)
        y_p = np.argmax(pred,-1)
        print('acc={}'.format(accuracy_score(y_t, y_p)))  
        print('mcc = {}'.format(matthews_corrcoef(y_t, y_p)))
        
        y_pred = np.concatenate((y_pred, pred))
        y_targ = np.concatenate((y_targ, y_test))  
        
    y_t = np.argmax(y_targ,-1)
    y_p = np.argmax(y_pred,-1)
    
    print('acc={}'.format(accuracy_score(y_t, y_p)))  
    print('mcc = {}'.format(matthews_corrcoef(y_t, y_p)))
    
def predictor2():
    (X_train_pos_ls, X_test_pos_ls), (X_train_neg_ls, X_test_neg_ls) = splitDataKF()
    y_pred, y_targ = np.zeros((0,2)), np.zeros((0,2))
    for i in range(5):
        # formulate protein seqs
        x_train_pos = protsFormulateByChaosCode(X_train_pos_ls[i])
        x_test_pos = protsFormulateByChaosCode(X_test_pos_ls[i])
        x_train_neg = protsFormulateByChaosCode(X_train_neg_ls[i])
        x_test_neg = protsFormulateByChaosCode(X_test_neg_ls[i])
        
        # bulid samples set and their labels
        x_test = np.concatenate((x_test_pos, x_test_neg))
        x_test = x_test.reshape(x_test.shape[0],2*ws+1,21,1)
        
        y_test = np.zeros((len(x_test), 2))
        y_test[:len(x_test_pos), 1] = 1 #正样本标记
        y_test[len(x_test_pos):,0] = 1 # 负样本标记
        
        # label positive samples
        y_pos = np.zeros((len(x_train_pos), 2))
        y_pos[:,1] = 1
        
        # trackle imbalanced dataset
        num_train_pos = x_train_pos.shape[0]
        num_train_neg = x_train_neg.shape[0]
        m = num_train_neg//num_train_pos
        ls_pred = []
        for j in range(m):#constructe m predictors
            # extract samples from negative set as the numbers as postive samples
            index = np.random.choice(num_train_neg, num_train_pos, replace=False)
            x_neg = x_train_neg[index]
            y_neg = np.zeros((len(x_neg),2))
            y_neg[:,0] = 1
            x_train = np.concatenate((x_train_pos, x_neg))
            y_train = np.concatenate((y_pos,y_neg))
            
            x_train, y_train = shuffle(x_train, y_train)
            x_train = x_train.reshape(x_train.shape[0],2*ws+1,21,1)
            model_name = 'keras_pdna224_trained_5fold_model_{}_{}.h5'.format(i,j)
            
            # train cnn and predict
            pred = net2(x_train, x_test, y_train, save_dir, model_name, batch_size, epochs)
            
            ls_pred.append(pred)
            
        arry_pred = np.array(ls_pred)
        arry_pred = np.average(arry_pred, axis=0)
                
        y_t = np.argmax(y_test,-1)
        y_p = np.argmax(arry_pred,-1)
        # metrics the predictor
        print('acc={}'.format(accuracy_score(y_t, y_p)))  
        print('mcc = {}'.format(matthews_corrcoef(y_t, y_p)))
        
        y_pred = np.concatenate((y_pred, arry_pred))
        y_targ = np.concatenate((y_targ, y_test))  

    # 5-kf metrics
    y_t = np.argmax(y_targ,-1)
    y_p = np.argmax(y_pred,-1)
    
    print('acc={}'.format(accuracy_score(y_t, y_p)))  
    print('mcc = {}'.format(matthews_corrcoef(y_t, y_p)))
            
predictor2()            
            
    
    
    
    
    
    
    