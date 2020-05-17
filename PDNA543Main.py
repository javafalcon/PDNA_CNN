# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:22:31 2020

@author: lwzjc
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, backend
from Capsule import CapsuleLayer, squash, Length, Mask, margin_loss
from tensorflow.keras.utils import to_categorical   
from numpy.random import default_rng
from sklearn.metrics import accuracy_score, matthews_corrcoef,confusion_matrix
   
def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          name='primaryCap_conv2d')(inputs)
    dim = output.shape[1]*output.shape[2]*output.shape[3]
    outputs = layers.Reshape(target_shape=(dim//dim_vector,dim_vector), name='primaryCap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


def CapsNet(input_shape, n_class, num_routing):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='valid',
                         activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=64, kernel_size=5, 
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



def trainAndTest(model, data, lr, lam_recon, batch_size, epochs):
    (x_train, y_train),(x_test, y_test) = data
    
    model.compile(optimizer=optimizers.Adam(lr=lr),
                 loss=[margin_loss, 'mse'],
                 loss_weights=[1., lam_recon],
                 metrics={'out_caps': 'accuracy'})
    
    # callbacks
    #log = callbacks.CSVLogger('./result/PDNA-543/log.csv')
    #tb = callbacks.TensorBoard(log_dir='./result/PDNA-543/tensorboard-logs',
    #                           batch_size=batch_size, histogram_freq=1)
    #checkpoint = callbacks.ModelCheckpoint('./result/PDNA-543/weights-{epoch:02d}.h5',
    #                                       save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))
    
    model.fit([x_train, y_train], [y_train, x_train],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[[x_test,y_test],[y_test,x_test]],
              callbacks=[lr_decay])
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=batch_size)
    y = np.argmax(y_pred, 1)
    return y

def randomDeepNets(model, data, nNet=50, seed=12345,
                   lr=0.001, lam_recon=0.3,
                   batch_size=50, epochs=5):
    (train_X, y_train), (test_X, y_test) = data
    # train_X.shape=[None,21,553]
    y_pred = np.zeros((len(y_test)),)
    
    rg = default_rng(seed)
    arr = np.arange(553)
    rg.shuffle(arr)
    
    for i in range(nNet):
        print("----------Training No.{} net---------".format(i))
        x_train = train_X[:,:, arr[i*21: (i+1)*21]]
        x_test = test_X[:,:, arr[i*21: (i+1)*21]]
        x_train = x_train.reshape(-1,21,21,1)
        x_test = x_test.reshape(-1,21,21,1)
        
        y = trainAndTest(model, ((x_train, y_train), (x_test, y_test)), 
                         lr, lam_recon, batch_size, epochs)
        y_pred += y
        backend.clear_session()
    return y_pred/nNet

def lr_schedule(epoch):
    lr = 1e-3
    return lr*0.9*epoch

if __name__ == "__main__":
    data = np.load('PDNA_543_onehot_10.npz')
    train_X, train_y = data['train_X'], data['train_y']
    test_X, test_y = data['test_X'], data['test_y'] 
    
    train_X = train_X.reshape(-1,21,20,1)
    test_X = test_X.reshape(-1,21,20,1)
    y_train = to_categorical(train_y, num_classes=2)
    y_test = to_categorical(test_y, num_classes=2)
    from resnet import resnet_v1
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
    import os
    model = resnet_v1(input_shape=[21,20,1],depth=20, num_classes=2)
    model.summary()
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                   min_lr=0.5e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  metrics=['accuracy'])
    
    save_dir = os.path.join(os.getcwd(), 'save_models')
    model_name = 'mnist_resnet_model.{epoch:02d}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)    
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc',
                                verbose=1, save_best_only=True)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    
    model.fit(train_X, y_train,
              batch_size=100,
              epochs=20,
              validation_data=(test_X,y_test),
              shuffle=True,
              callbacks=callbacks)
    
    y_pred = model.predict(test_X)
    
    y_pred = np.argmax(y_pred, 1)
    
    '''
    model = CapsNet(input_shape=[21,20,1], n_class=2, num_routing=3)
    model.summary()
    y_p = randomDeepNets(model, ((train_X, y_train), (test_X, y_test)), 
                            nNet=50, seed=12345, batch_size=100)
    y_pred = (y_p>0.5).astype(float)
    
    y_pred = trainAndTest(model, ((train_X, y_train), (test_X, y_test)),
                     lr=0.001, lam_recon=0.35, batch_size=100, epochs=10)
    '''
    print('Test Accuracy:', accuracy_score(test_y, y_pred))
    print('Test mattews-corrcoef', matthews_corrcoef(test_y, y_pred))
    print('Test confusion-matrix', confusion_matrix(test_y, y_pred))






