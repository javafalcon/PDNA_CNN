# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:07:00 2020

@author: lwzjc
"""

from prepareData import readPDNA543_seqs_sites
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from resnet import resnet_v1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
import os

from Capsule import CapsuleLayer, squash, Length, Mask, margin_loss
from tensorflow.keras.utils import to_categorical   

from sklearn.metrics import accuracy_score, matthews_corrcoef,confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from util import pcaval

from imblearn.over_sampling import ADASYN 

row, col, step = 15, 21, 8
def slipSeqs(seqs:dict, sites:dict, windown_size=10, step=7): 
    # 按步长step滑窗，滑窗大小windown_size
    # 使得相邻的两个序列片段重复windown_size - step个氨基酸
    # 如果序列片段中至少有一个结合位点，则为正样本集；否则为负样本集
    # 对于序列最右则，若氨基酸个数不足window_size，则从最右则向左取window_size个氨基酸
    pos_seqs = []
    neg_seqs = []
    targets = []
    for key in seqs.keys():
        seq = seqs[key]
        site = sites[key]
        n = len(seq)
        for i in range(0, n, step):
            if i + windown_size > n:
                start = n - windown_size
                end = n
            else:
                start = i
                end = i + windown_size
                
            if '1' in site[start:end]:
                pos_seqs.append( seq[start:end])
                targets.append( site[start:end])
            else:
                neg_seqs.append( seq[start:end])
                
    return (pos_seqs, targets, neg_seqs)


def onehotSeqs(seqs):
    amino_acids = "ACDEFHIGKLMNQPRSTVWY"
    X = list()
    for seq in seqs:
        x = np.zeros(shape=(row,col))   
        for i in range(len(seq)):
            try:
                k = amino_acids.index(seq[i])
                x[i,k] = 1
            except ValueError:
                x[i] = np.ones((col,))*0.05
        X.append(x)
    return np.array(X)

def load_TrainData(random_state):
    data = np.load('PDNA_Data\\PDNA_543_train_7.npz')
    train_pos_seqs = data['pos'] 
    train_neg_seqs = data['neg']
    #train_pos_X = onehotSeqs(train_pos_seqs)
    #train_neg_X = onehotSeqs(train_neg_seqs)
    train_pos_X = pcaval(train_pos_seqs)
    train_neg_X = pcaval(train_neg_seqs)
    train_neg_X = shuffle(train_neg_X)
    
    X_train = np.concatenate((train_pos_X, train_neg_X[:int(len(train_pos_X)*3)]))
    y_train = np.zeros((len(X_train),))
    y_train[:len(train_pos_X)] = 1.

    ada = ADASYN(random_state=random_state)
    X_train = X_train.reshape((-1,15*21))
    X_res, y_res = ada.fit_resample(X_train, y_train)
    X_res = X_res.reshape((-1,15,21))
    return (X_res, y_res)

def load_resample_TrainData():
    data = np.load('PDNA_Data\\PDNA_543_train_7.npz')
    train_pos_seqs = data['pos'] 
    train_neg_seqs = data['neg']
    train_pos_X = pcaval(train_pos_seqs)
    train_neg_X = pcaval(train_neg_seqs)
    train_neg_X = shuffle(train_neg_X)
    
    X_train = np.concatenate((train_pos_X, train_neg_X))
    y_train = np.zeros((len(X_train),))
    y_train[:len(train_pos_X)] = 1.
    
    ada = ADASYN(random_state=42)
    X_train = X_train.reshape((-1,row*col))
    X_res, y_res = ada.fit_resample(X_train, y_train)
    X_res = X_res.reshape((-1,row,col))
    return (X_res, y_res)

def load_TestData():
    data = np.load('PDNA_Data\\PDNA_543_test_7.npz')
    test_pos_seqs = data['pos'] 
    test_neg_seqs = data['neg']
    #test_pos_X = onehotSeqs(test_pos_seqs)
    #test_neg_X = onehotSeqs(test_neg_seqs)
    test_pos_X = pcaval(test_pos_seqs)
    test_neg_X = pcaval(test_neg_seqs)
    X_test = np.concatenate((test_pos_X, test_neg_X))
    y_test = np.zeros((len(X_test),))
    y_test[:len(test_pos_X)] = 1.
    
    return (X_test, y_test)

def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    '''output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          name='primaryCap_conv2d')(inputs)
    dim = output.shape[1]*output.shape[2]*output.shape[3]'''
    output = layers.Conv1D(filters=dim_vector*n_channels, kernel_size=kernel_size,
                           strides=strides, padding=padding,
                           name='primaryCap_conv1d')(inputs)
    dim = output.shape[1]*output.shape[2]
    outputs = layers.Reshape(target_shape=(dim//dim_vector,dim_vector), name='primaryCap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


def CapsNet(input_shape, kerl, n_class, num_routing):
    x = layers.Input(shape=input_shape)
    '''conv1 = layers.Conv2D(filters=256, kernel_size=kerl, strides=1, padding='valid',
                         activation='relu', name='conv1')(x)
    
    conv2 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                         activation='relu', name='conv2')(conv1)
    conv3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid',
                         activation='relu', name='conv3')(conv2)
    '''
    conv1 = layers.Conv1D(filters=256, kernel_size=kerl, strides=1, padding='same',
                          activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=kerl, 
                            strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing,
                            name='digitcaps')(primarycaps)
    out_caps = Length(name='out_caps')(digitcaps)
    
    # Decoder network
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(np.prod(input_shape), activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)
    
    return models.Model([x,y], [out_caps, x_recon])

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

def train_test(X_train, y_train, X_test, y_test, model, model_name):
    tf.keras.backend.clear_session()
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                   min_lr=0.5e-6)

    save_dir = os.path.join(os.getcwd(), 'save_models')
    model_name = model_name + '.{epoch:02d}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)    
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc',
                                verbose=1, save_best_only=True,save_weights_only=True,)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    '''
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  metrics=['accuracy'])
    
    model.load_weights(model_name)
    
    model.fit(X_train, y_train, batch_size=100, epochs=20, 
              validation_data=[[X_test, y_test], [y_test, X_test]],
              shuffle=True,
              callbacks=callbacks)
    '''
    
    # compile just for capsul
    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule(0)),
             loss=[margin_loss, 'mse'],
             loss_weights=[1., 0.325],
             metrics={'out_caps': 'accuracy'})
    
    
    # fit just for capsul
    model.fit([X_train, y_train], [y_train, X_train],
              batch_size=100,
              epochs=15,
              validation_data=[[X_test, y_test], [y_test, X_test]],
              shuffle=True,
              callbacks=callbacks)
        
    pred, _ = model.predict([X_test, y_test])
    
    y_pred = np.argmax(pred, 1)
    
    return y_pred

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

def ensembler():
    (X_test, test_y) = load_TestData()
    y_pred = np.zeros(shape=(test_y.shape[0],))
    ls_pred = []
    kerls = [9,7,7,5,5]
    random_states = [24,42,57,136,256]
    K = 5
    for (kerl, random_state) in zip(kerls, random_states):
        (X_train, train_y) = load_TrainData(random_state)
            
        #train_X = X_train.reshape(-1,row,col,1)
        #test_X = X_test.reshape(-1,row,col,1)
        y_train = to_categorical(train_y, num_classes=2)
        y_test = to_categorical(test_y, num_classes=2)
        tf.keras.backend.clear_session()
        model = CapsNet(input_shape=[row,col],kerl=kerl, n_class=2, num_routing=5)
        #model = CnnNet(input_shape=[row,col,1], n_class=2)
        #model = resnet_v1(input_shape=[row,col,1],depth=20, num_classes=2)
        model.summary()
        
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                       min_lr=0.5e-6)
        '''
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=lr_schedule(0)),
                      metrics=['accuracy'])
        '''
        
        # compile just for capsul
        model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule(0)),
                 loss=[margin_loss, 'mse'],
                 loss_weights=[1., 0.325],
                 metrics={'out_caps': 'accuracy'})
        
        save_dir = os.path.join(os.getcwd(), 'save_models')
        model_name = 'capsul_model.{epoch:02d}.h5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)    
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc',
                                    verbose=1, save_best_only=True)
        callbacks = [checkpoint, lr_reducer, lr_scheduler]
        
        # fit just for capsul
        model.fit([X_train, y_train], [y_train, X_train],
                  batch_size=100,
                  epochs=15,
                  validation_data=[[X_test, y_test], [y_test, X_test]],
                  shuffle=True,
                  callbacks=callbacks)
        
        pred, _ = model.predict([X_test, y_test])
        
        y_pred += np.argmax(pred, 1)
        
        ls_pred.append(y_pred)
        
    y_pred = y_pred/K
    y_pred = (y_pred>0.5).astype(float)
    print('Test Accuracy:', accuracy_score(test_y, y_pred))
    print('Test mattews-corrcoef', matthews_corrcoef(test_y, y_pred))
    print('Test confusion-matrix', confusion_matrix(test_y, y_pred))   
           
if __name__ == "__main__":
    K = 9
    (X_test, test_y) = load_TestData()
    y_test = to_categorical(test_y, num_classes=2)
    y_pred = np.zeros(shape=(test_y.shape[0],))
    
    model = CapsNet(input_shape=[row,col],kerl=5, n_class=2, num_routing=5)
    X, y = load_resample_TrainData()
    pos_index, neg_index = [], []
    for i in range(len(y)):
        if y[i] == 1:
            pos_index.append(i)
        else:
            neg_index.append(i)
            
    X_pos, y_pos = X[pos_index], y[pos_index]
    X_neg, y_neg = X[neg_index], y[neg_index]
    
    X_pos, y_pos = shuffle(X_pos, y_pos)
    X_neg, y_neg = shuffle(X_neg, y_neg)
    
    posN, negN = X_pos.shape[0], X_neg.shape[0]
    posM, negM = posN//K, negN//K
    for i in range(K):
        ps, pe = i*posM, (i+1)*posM
        if pe > posN:
            pe = posN
        
        ns, ne = i*negM, (i+1)*negM
        if ne > negN:
            ne = negN
        
        X_train = np.concatenate((X_pos[ps:pe], X_neg[ns:ne]))
        y_train = np.concatenate((y_pos[ps:pe], y_neg[ns:ne]))
        y_train = to_categorical(y_train, num_classes=2)
        
        pred = train_test(X_train, y_train, X_test, y_test, model, 'capsul_ens{}'.format((i)))
        
        y_pred += pred
        
    y_pred = y_pred/K
    y_pred = (y_pred>0.5).astype(float)
    print('Test Accuracy:', accuracy_score(test_y, y_pred))
    print('Test mattews-corrcoef', matthews_corrcoef(test_y, y_pred))
    print('Test confusion-matrix', confusion_matrix(test_y, y_pred))
    

                
                
                
                
                
                
                