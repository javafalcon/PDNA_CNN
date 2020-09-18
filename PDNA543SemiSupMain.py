# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:16:58 2020

@author: lwzjc
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from resnet import resnet_v1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical   

from sklearn.metrics import accuracy_score, matthews_corrcoef,confusion_matrix
from sklearn.utils import shuffle
from util import pcaval

from imblearn.over_sampling import ADASYN 

from configparser import ConfigParser
from semisupLearner_keras import displayMetrics
from semisupLearner_keras import SemisupLearner

def load_downsample_TrainData(df, alpa=1, random_state=None):
    data = np.load(df)#'PDNA_Data\\PDNA_543_train_7.npz'
    train_pos_seqs = data['pos'] 
    train_neg_seqs = data['neg']
 
    train_pos_X = pcaval(train_pos_seqs)
    train_neg_X = pcaval(train_neg_seqs)
    train_neg_X = shuffle(train_neg_X, random_state=random_state)
    
    X_train = np.concatenate((train_pos_X, train_neg_X[:int(len(train_pos_X)*alpa)]))
    y_train = np.zeros((len(X_train),))
    y_train[:len(train_pos_X)] = 1.

    return (X_train, y_train)

def load_upsample_TrainData(df, random_state=None):
    data = np.load(df)#'PDNA_Data\\PDNA_543_train_7.npz'
    train_pos_seqs = data['pos'] 
    train_neg_seqs = data['neg']
    train_pos_X = pcaval(train_pos_seqs)
    train_neg_X = pcaval(train_neg_seqs)
    train_neg_X = shuffle(train_neg_X)
    
    X_train = np.concatenate((train_pos_X, train_neg_X))
    y_train = np.zeros((len(X_train),))
    y_train[:len(train_pos_X)] = 1.
    
    ada = ADASYN(random_state=random_state)
    X_train = X_train.reshape((-1,row*col))
    X_res, y_res = ada.fit_resample(X_train, y_train)
    X_res = X_res.reshape((-1,row,col))
    return (X_res, y_res)

def load_updownsample_TrainData(df, alpa=3, random_state=None):
    data = np.load(df)#'PDNA_Data\\PDNA_543_train_7.npz'
    train_pos_seqs = data['pos'] 
    train_neg_seqs = data['neg']
    train_pos_X = pcaval(train_pos_seqs)
    train_neg_X = pcaval(train_neg_seqs)
    train_neg_X = shuffle(train_neg_X)
    
    X_train = np.concatenate((train_pos_X, train_neg_X[:int(len(train_pos_X)*alpa)]))
    y_train = np.zeros((len(X_train),))
    y_train[:len(train_pos_X)] = 1.

    ada = ADASYN(random_state=random_state)
    X_train = X_train.reshape((-1,row*col))
    X_res, y_res = ada.fit_resample(X_train, y_train)
    X_res = X_res.reshape((-1,row,col))
    return (X_res, y_res)
    
def load_TestData(df):
    data = np.load(df)#'PDNA_Data\\PDNA_543_test_7.npz'
    test_pos_seqs = data['pos'] 
    test_neg_seqs = data['neg']
     
    test_pos_X = pcaval(test_pos_seqs)
    test_neg_X = pcaval(test_neg_seqs)
    X_test = np.concatenate((test_pos_X, test_neg_X))
    y_test = np.zeros((len(X_test),))
    y_test[:len(test_pos_X)] = 1.
    
    return (X_test, y_test)

def CnnNet(input_shape, num_classes):
    regular = keras.regularizers.l1(0.01)
    x = keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (5,5), strides=1,
                          padding='same', activation='relu', 
                          kernel_regularizer=regular,
                          name='conv1')(x)
    conv2 = layers.Conv2D(32, (5,5), padding='same', activation='relu', name='conv2')(conv1)
    pool1 = layers.MaxPool2D(pool_size=(2,2))(conv2)
    drop1 = layers.Dropout(0.25)
    x_a = drop1(pool1)
    x_b = drop1(pool1)
    
    conv3 = layers.Conv2D(64,(5,5), padding='same',
                          activation='relu', name='conv3')
    conv4 = layers.Conv2D(128, (5,5), activation='relu', name='conv4')
    pool2 = layers.MaxPool2D()
    drop2 = layers.Dropout(0.25)
    
    x_a = conv3(x_a)
    x_a = conv4(x_a)
    x_a = pool2(x_a)
    x_a = drop2(x_a)
    
    x_b = conv3(x_b)
    x_b = conv4(x_b)
    x_b = pool2(x_b)
    x_b = drop2(x_b)
    flat = layers.Flatten()
    dens1 = layers.Dense(512, activation='relu')
    drop3 = layers.Dropout(0.5)
    out = layers.Dense(num_classes, activation='softmax', name="unsupLayer")
    
    x_a = flat(x_a)
    x_a = dens1(x_a)
    x_a = drop3(x_a)
    out_a = out(x_a)
    
    x_b = flat(x_b)
    x_b = dens1(x_b)
    x_b = drop3(x_b)
    out_b = out(x_b)
    
    return keras.Model(x, [out_a, out_b]) 

def semiNet(input_shape, num_classes):
    l2value = 0.001
    regular = tf.keras.regularizers.l1(0.001)
    x_input = layers.Input(shape=input_shape,name="main_input")
    conv1 = layers.Conv2D(32,(9,1),padding='same',activation="relu",kernel_regularizer=regular)(x_input) 
    pool1 = layers.MaxPooling2D(pool_size=(7,1))(conv1)
    drop1 = layers.Dropout(0.3)
    
    #conv2 = Conv2D(64,(3,3),padding='same',activation="relu",kernel_regularizer=l2(l2value))(conv1)               
    
    #conv3 = Conv2D(100,5,padding='same',activation="relu",kernel_regularizer=l2(l2value))(conv2) 
    
    #pool1 = MaxPooling2D(2)(conv3)
    
    #drop_1 = Dropout(0.3)
    x_a = drop1(pool1)
    x_b = drop1(pool1)
        
    flatten = layers.Flatten()
    x_a = flatten(x_a)
    x_b = flatten(x_b)
        
    dense_1 = layers.Dense(100, activation='relu')
    x_a = dense_1(x_a)
    x_b = dense_1(x_b)
    
    drop_3 = layers.Dropout(0.3, name="unsupLayer")
    x_a = drop_3(x_a)
    x_b = drop_3(x_b)
    
    out = layers.Dense(num_classes, activation="softmax")(x_a)
    
    model = models.Model(inputs=x_input, outputs=[out, x_b])
    return model

def lr_schedule(epoch):
    lr = 1e-3
    return lr*0.9*epoch

def mylos(o1, o2):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + tf.keras.losses.mean_squared_error(y_true, y_pred)
def train_test(X_train, y_train, X_test, y_test, model, modelFile, noteInfo, metricsFile, **confParam):
    tf.keras.backend.clear_session()
    
    ssparam = {}
    ssparam['x_train'] = X_train
    ssparam['y_train'] = [y_train, y_train]
    ssparam['x_vldt'] = X_test
    ssparam['y_vldt'] = [y_test, y_test]
    ssparam['batch_size'] = confParam['batch_size']
    ssparam['epochs'] = confParam['epochs']
    ssparam['patience'] = confParam['patience']
    ssparam['rampup_length'] = confParam['rampup_length']
    ssparam['rampdown_length'] = confParam['rampdown_length']
    ssparam['learning_rate_max'] = confParam['learning_rate_max']
    ssparam['scaled_unsup_weight_max'] = confParam['scaled_unsup_weight_max']
    ssparam['gammer'] = confParam['gammer']
    ssparam['beita'] = confParam['beita']
    ssparam['learning_rate'] = confParam['learning_rate']
    ssl = SemisupLearner(modelFile, model, **ssparam)
    ssl.train()
    pred_prob = ssl.predict(X_test)
    displayMetrics(y_test, pred_prob, noteInfo, metricsFile)
    return pred_prob


def ensemble_train_test(X_trains, y_trains, X_test, y_test, model, K):
    
    y_pred = np.zeros(shape=(y_test.shape[0],))
    ls_pred = []
        
    for i in range(K):
        pred = train_test(X_trains[i], y_trains[i], X_test, y_test, model, )
        p = np.argmax(pred, 1)
        y_pred += np.argmax(pred, 1)
        
        ls_pred.append(p)
    y_pred = y_pred/K
    y_pred = (y_pred>0.5).astype(float)    
    return y_pred, ls_pred

def downsample_ensembler(trainFile, testFile, metricsFile, K, alpa, **confParam):   
    row, col = confParam['row'], confParam['col']
    
    (X_test, test_y) = load_TestData(testFile)
    X_test = X_test.reshape(-1,row,col,1)
    y_test = to_categorical(test_y, num_classes=2)
    
    y_pred = np.zeros(shape=(test_y.shape[0],))
    ls_pred = []
    
    random_state = [42,163,1436,57,342,93,3987,1135,198743]   
    
    for i in range(K):
        (X_train, train_y) = load_downsample_TrainData(trainFile, alpa, random_state[i])
            
        X_train = X_train.reshape(-1,row,col,1)
        y_train = to_categorical(train_y, num_classes=2)
        
        keras.backend.clear_session()
        model = CnnNet(input_shape=[row,col,1],num_classes=2)
        #model = semiNet(input_shape=[row,col,1],num_classes=2)
        model.summary()
        
        noteInfo = "Ensemble {}/{} semisuper learners on downsample dataset \
                    (Numbers of negative = {} * Numbers of postive".format(i, K, alpa)
        modelFile = "save_models/ensem_semisup_downsample_{}.hdf5".format(i)
        pred = train_test(X_train, y_train, X_test, y_test, model, modelFile, 
                          noteInfo, metricsFile, **confParam)
        y_pred += np.argmax(pred, 1)
        
        ls_pred.append(y_pred)
        
    y_pred = y_pred/K
    y_pred = (y_pred>0.5).astype(float)
    print('Test Accuracy:', accuracy_score(test_y, y_pred))
    print('Test mattews-corrcoef', matthews_corrcoef(test_y, y_pred))
    print('Test confusion-matrix', confusion_matrix(test_y, y_pred))   
           
def updownsample_ensembler(trainFile, testFile, metricsFile,  K, alpa, **confParam):   
    row, col = confParam['row'], confParam['col']
    (X_test, test_y) = load_TestData(testFile)
    X_test = X_test.reshape(-1,row,col,1)
    y_test = to_categorical(test_y, num_classes=2)
    
    y_pred = np.zeros(shape=(test_y.shape[0],))
    ls_pred = []
    
    random_state = [42,163,1436,57,342,93,3987,1135,198743]   
    
    for i in range(K):
        (X_train, train_y) = load_updownsample_TrainData(trainFile, alpa, random_state[i])
            
        X_train = X_train.reshape(-1,row,col,1)
        y_train = to_categorical(train_y, num_classes=2)
        
        tf.keras.backend.clear_session()
        model = CnnNet(input_shape=[row,col,1],num_classes=2)
        model.summary()
        
        noteInfo = "Ensemble {}/{} semisuper learners on up & downsample dataset \
                    (Numbers of negative = {} * Numbers of postive".format(i, K, alpa)
        modelFile = "save_models/ensem_semisup_updownsample_{}.hdf5".format(i)
        pred = train_test(X_train, y_train, X_test, y_test, model, modelFile, 
                          noteInfo, metricsFile, **confParam)
        y_pred += np.argmax(pred, 1)
        
        ls_pred.append(y_pred)
        
    y_pred = y_pred/K
    y_pred = (y_pred>0.5).astype(float)
    print('Test Accuracy:', accuracy_score(test_y, y_pred))
    print('Test mattews-corrcoef', matthews_corrcoef(test_y, y_pred))
    print('Test confusion-matrix', confusion_matrix(test_y, y_pred))  
    
def readConfParam():
    conf = ConfigParser()
    conf.read('conf.ini')
    confParam = {}
    confParam['row'] = int(conf.get('netparam', 'row'))
    confParam['col'] = int(conf.get('netparam', 'col'))
    confParam['batch_size'] = int(conf.get('netparam','batch_size'))
    confParam['epochs'] = int(conf.get('netparam', 'epochs'))
    confParam['patience'] = int(conf.get('netparam', 'patience'))
    confParam['learning_rate'] = float(conf.get('netparam', 'learning_rate'))
    
    confParam['rampup_length'] = int(conf.get('semisupparam', 'rampup_length'))
    confParam['rampdown_length'] = int(conf.get('semisupparam', 'rampdown_length'))
    confParam['learning_rate_max'] = float(conf.get('semisupparam', 'learning_rate_max'))
    confParam['scaled_unsup_weight_max'] = int(conf.get('semisupparam', 'scaled_unsup_weight_max'))
    confParam['gammer'] = float(conf.get('semisupparam', 'gammer'))
    confParam['beita'] = float(conf.get('semisupparam', 'beita'))
    
    return confParam

if __name__ == "__main__":
    trainFile = 'PDNA_543_train_7.npz'
    testFile= 'PDNA_543_test_7.npz'
    confParam = readConfParam()
    metricsFile = "ensemble_semisup_info.txt"
    downsample_ensembler(trainFile, testFile, metricsFile, 9, 3, **confParam)