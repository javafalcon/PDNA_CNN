# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:56:30 2020

@author: lwzjc
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from resnet import resnet_v1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical   

from sklearn.metrics import accuracy_score, matthews_corrcoef,confusion_matrix
from sklearn.utils import shuffle
from util import pcaval

from imblearn.over_sampling import ADASYN 

from configparser import ConfigParser

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
    
    ada = ADASYN(random_state=seed)
    X_train = X_train.reshape((-1,row*col))
    X_res, y_res = ada.fit_resample(X_train, y_train)
    X_res = X_res.reshape((-1,row,col))
    return (X_res, y_res)

def load_updownsample_TrainData(df, alpa=3, seed=random_state):
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

def CnnNet(input_shape, n_class):
    regular = tf.keras.regularizers.l1(0.01)
    x = layers.Input(shape=input_shape)
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
    out = layers.Dense(n_class, activation='softmax')(drop3)
    
    return models.Model(x, out) 

def lr_schedule(epoch):
    lr = 1e-3
    return lr*0.9*epoch

def train_test(X_train, y_train, X_test, y_test, model, modelFile, **confParam):
    tf.keras.backend.clear_session()
    
    batch_size = confParam['batch_size']
    epochs = confParam['epochs']
    patience = confParam['patience']
    learning_rate = confParam['learning_rate']
    #rampup_length = confParam['rampup_length']
    #rampdown_length = confParam['rampdown_length']
    #learning_rate_max = confParam['learning_rate_max']
    #scaled_unsup_weight_max = confParam['scaled_unsup_weight_max']
    #gammer = confParam['gammer']
    #beita = confParam['beita']  
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy']
                  )
    
    # callbacks
    log = callbacks.CSVLogger('./result/PDNA-543/log.csv')
        
    checkpoint = callbacks.ModelCheckpoint(modelFile,
                                           monitor='val_loss',
                                           save_best_only=True, 
                                           save_weights_only=True, 
                                           verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: learning_rate * (0.9 ** epoch))
    earlystop = callbacks.EarlyStopping(patience=patience, monitor='val_loss')
    
    cbs=[log,lr_decay]

    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
              validation_data=[X_test, y_test],
              shuffle=True,
              callbacks=cbs
              )
    
    pred_prob = model.predict(X_test)
    
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

def downsample_ensembler(alpa, **confParam):
    trainFile = 'PDNA_543_train_7.npz'
    testFile= 'PDNA_543_test_7.npz'
    
    (X_test, test_y) = load_TestData(testFile)
    X_test = X_test.reshape(-1,row,col,1)
    y_test = to_categorical(test_y, num_classes=2)
    
    y_pred = np.zeros(shape=(test_y.shape[0],))
   
    random_state = [42,163,1436,57,342,93,3987,1135,198743]
    K = 9
    for i in range(K):
        (X_train, train_y) = load_downsample_TrainData(trainFile, alpa, random_state[i])
            
        X_train = X_train.reshape(-1,row,col,1)
        y_train = to_categorical(train_y, num_classes=2)
        
        modelFile = "./result/PDNA-543/weight-{}.h5".format(random_state[i])
        tf.keras.backend.clear_session()
        #model = CnnNet(input_shape=[row,col,1],num_classes=2)
        model = resnet_v1(input_shape=[row,col,1],depth=20, num_classes=2)
        model.summary()
        #model.load_weights(modelFile)
        #pred = model.predict(X_test)
        pred = train_test(X_train, y_train, X_test, y_test, model, modelFile, **confParam)
        y_pred += np.argmax(pred, 1)
               
    y_pred = y_pred/K
    y_pred = (y_pred>0.5).astype(float)
    print('Test Accuracy:', accuracy_score(test_y, y_pred))
    print('Test mattews-corrcoef', matthews_corrcoef(test_y, y_pred))
    print('Test confusion-matrix', confusion_matrix(test_y, y_pred))   
           
def updownsample_ensembler(alpa, **confParam):
    trainFile = 'PDNA_543_train_7.npz'
    testFile= 'PDNA_543_test_7.npz'
    
    K = 9
    (X_test, test_y) = load_TestData(testFile)
    X_test = X_test.reshape(-1,row,col,1)
    y_test = to_categorical(test_y, num_classes=2)
    y_pred = np.zeros(shape=(test_y.shape[0],))
    
    random_state = [42,163,1436,57,342,93,3987,1135,198743]
    for i in range(K):
        (X_train, train_y) = load_updownsample_TrainData(trainFile, seed=random_state[i], alpa=alpa)
        X_train = X_train.reshape(-1,row,col,1)
        y_train = to_categorical(train_y, num_classes=2)
        
        modelFile = "./result/PDNA-543/updownsample-weight-{}.h5".format(random_state[i])
        tf.keras.backend.clear_session()
        #model = CnnNet(input_shape=[row,col,1],num_classes=2)
        model = resnet_v1(input_shape=[row,col,1],depth=20, num_classes=2)
        model.summary()
        #model.load_weights(modelFile)
        #pred = model.predict(X_test)
        pred = train_test(X_train, y_train, X_test, y_test, model, modelFile, **confParam)
        y_pred += np.argmax(pred, 1)
        
    y_pred = y_pred/K
    y_pred = (y_pred>0.5).astype(float)
    print('Test Accuracy:', accuracy_score(test_y, y_pred))
    print('Test mattews-corrcoef', matthews_corrcoef(test_y, y_pred))
    print('Test confusion-matrix', confusion_matrix(test_y, y_pred))
    
def readConfParam():
    conf = ConfigParser()
    conf.read('conf.ini')
    confParam = {}
    confParam['row'] = int(conf.get('negparam', 'row'))
    confParam['col'] = int(conf.get('negparam', 'col'))
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
    confParam = readConfParam()
    
    updownsample_ensembler(2, **confParam)