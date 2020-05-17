# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:18:02 2020

@author: lwzjc
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical   
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, matthews_corrcoef,confusion_matrix
from sklearn.utils import resample 

from Capsule import CapsuleLayer, squash, Length, Mask, margin_loss
from resnet import resnet_v1, lr_schedule
from dataset543 import gen_PDNA543_HHM,readPDNA543_hhm_sites

def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          name='primaryCap_conv2d')(inputs)
    dim = output.shape[1]*output.shape[2]*output.shape[3]
    outputs = layers.Reshape(target_shape=(dim//dim_vector,dim_vector), name='primaryCap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


def CapsNet(input_shape, n_class, num_routing):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                         activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=64, kernel_size=5, 
                            strides=2, padding='same')
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

def load_PDNA543_hhm_train(train_hhm, train_sites, ws=15):   
    #(train_hhm, train_sites), (test_hhm, test_sites) = readPDNA543_hhm_sites()   
    traindatafile = 'PDNA543_HHM_{}.npz'.format(ws)
    
    x_train_pos, x_train_neg = gen_PDNA543_HHM(train_hhm, train_sites, traindatafile, ws=ws)
    
    x_neg = resample(x_train_neg, n_samples=x_train_pos.shape[0], replace=False)
    x_train = np.concatenate((x_train_pos, x_neg))
    y_train = np.zeros((x_train.shape[0],))
    y_train[:x_train_pos.shape[0]] = 1          
    return (x_train, y_train)

def load_PDNA543_hhm_test(test_hhm, test_sites, ws=15):   
    #(train_hhm, train_sites), (test_hhm, test_sites) = readPDNA543_hhm_sites()
    testdatafile = 'PDNA543TEST_HHM_{}.npz'.format(ws)
    
    x_test_pos, x_test_neg = gen_PDNA543_HHM(test_hhm, test_sites, testdatafile, ws=ws)
    x_test = np.concatenate((x_test_pos, x_test_neg))
    y_test = np.zeros((x_test.shape[0],))
    y_test[:x_test_pos.shape[0]] = 1            
    return (x_test, y_test)

if __name__ == "__main__":
    ws = 5
    row = 2*ws + 1
    col = 30
    batch_size = 48
    epochs = 20
    num_routing = 3
    
    (train_hhm, train_sites), (test_hhm, test_sites) = readPDNA543_hhm_sites() 
    (x_test, y_T) = load_PDNA543_hhm_test(test_hhm, test_sites, ws)
    x_test = x_test.reshape((-1,row,col,1))
    y_test = to_categorical(y_T.astype('float32'))    
    y_pred = np.zeros(shape=(y_test.shape[0],))
    
    # ensembling various networks
    # ensembling 3 resnets
    save_dir = os.path.join(os.getcwd(), 'save_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                       min_lr=0.5e-6)
    
    for _ in range(5):
        tf.keras.backend.clear_session()
        (x_train, y_train) = load_PDNA543_hhm_train(train_hhm, train_sites, ws)
        x_train = x_train.reshape((-1,row,col,1))
        y_train = to_categorical(y_train.astype('float32'))
        model = resnet_v1(input_shape=[row,col,1], depth=20, num_classes=2)
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=lr_schedule(0)),
                      metrics=['accuracy'])
        model.summary()
        
        # Prepare model model saving directory
        model_name = 'resnet_model.{epoch:02d}.h5'
        filepath = os.path.join(save_dir, model_name)    
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc',
                                    verbose=1)
        #callbacks = [checkpoint, lr_reducer, lr_scheduler]
        
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test,y_test),
                  shuffle=True)
        
        pred = model.predict([x_test, y_test], batch_size=batch_size)
        y_pred += np.argmax(pred, 1)
    
    # ensembling 3 capsul nets
    for _ in range(4):
        tf.keras.backend.clear_session()
        (x_train, y_train) = load_PDNA543_hhm_train(train_hhm, train_sites, ws)
        x_train = x_train.reshape((-1,row,col,1))
        y_train = to_categorical(y_train.astype('float32'))
        model = CapsNet(input_shape=x_train.shape[1:], n_class=2, num_routing=num_routing)
        model.summary()
        model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule(0)),
                      loss=[margin_loss,'mse'],
                      loss_weights=[1., 0.32],
                      metrics={'out_caps': "accuracy"})
        
        # 
        model_name = 'capsul_model.{epoch:02d}.h5'
        filepath = os.path.join(save_dir, model_name)  
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc',
                                    verbose=1)
        callbacks = [checkpoint, lr_reducer, lr_scheduler]

        model.fit([x_train, y_train], [y_train, x_train], 
                  batch_size=batch_size, 
                  epochs=epochs,
                  validation_data=[[x_test, y_test], [y_test, x_test]], 
                  shuffle=True)
        pred, recon = model.predict([x_test, y_test], batch_size=batch_size)
        y_pred += np.argmax(pred, 1)
    
    # Result 
    y_pred = y_pred/9
    y_P = (y_pred>0.25).astype(float)
    print('Test Accuracy:', accuracy_score(y_T, y_P))
    print('Test mattews-corrcoef', matthews_corrcoef(y_T, y_P))
    print('Test confusion-matrix', confusion_matrix(y_T, y_P))
    
    
    
    