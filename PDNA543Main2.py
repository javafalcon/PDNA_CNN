# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:07:00 2020

@author: lwzjc
"""

from prepareData import readPDNA543_seqs_sites
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, backend
from Capsule import CapsuleLayer, squash, Length, Mask, margin_loss
from tensorflow.keras.utils import to_categorical   
from numpy.random import default_rng
from sklearn.metrics import accuracy_score, matthews_corrcoef,confusion_matrix

row, col, step = 12, 20, 8
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

def load_data():
     # train_seqs, train_sites, test_seqs, test_sites是字典类型，key是蛋白质id
    (train_seqs, train_sites), (test_seqs, test_sites) = readPDNA543_seqs_sites()
    train_pos_seqs, train_targets, train_neg_seqs = slipSeqs(train_seqs, train_sites, row, step)
    test_pos_seqs, test_targets, test_neg_seqs = slipSeqs(test_seqs, test_sites, row, step)   
    train_pos_X = onehotSeqs(train_pos_seqs)
    train_neg_X = onehotSeqs(train_neg_seqs)
    X_train = np.concatenate((train_pos_X, train_neg_X))
    y_train = np.zeros((len(X_train),))
    y_train[:len(train_pos_X)] = 1.
    
    test_pos_X = onehotSeqs(test_pos_seqs)
    test_neg_X = onehotSeqs(test_neg_seqs)
    X_test = np.concatenate((test_pos_X, test_neg_X))
    y_test = np.zeros((len(X_test),))
    y_test[:len(test_pos_X)] = 1.
    
    return (X_train, y_train), (X_test, y_test), (train_targets, test_targets)

def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          name='primaryCap_conv2d')(inputs)
    dim = output.shape[1]*output.shape[2]*output.shape[3]
    outputs = layers.Reshape(target_shape=(dim//dim_vector,dim_vector), name='primaryCap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


def CapsNet(input_shape, n_class, num_routing):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid',
                         activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=3, 
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
              validation_data=[[x_test,y_test],[y_test,x_test]])
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=batch_size)
    '''
    model.compile(optimizer=optimizers.Adam(lr=lr), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_split=0.2)
    '''
    y_pred = model.predict(x_test)
    
    y = np.argmax(y_pred, 1)
    return y             

def lr_schedule(epoch):
    lr = 1e-3
    return lr*0.9*epoch

                
if __name__ == "__main__":
    (X_train, train_y), (X_test, test_y), (train_targets, test_targets) = load_data()
        
    train_X = X_train.reshape(-1,row,col,1)
    test_X = X_test.reshape(-1,row,col,1)
    y_train = to_categorical(train_y, num_classes=2)
    y_test = to_categorical(test_y, num_classes=2)
    tf.keras.backend.clear_session()
    #model = CapsNet(input_shape=[row,col,1], n_class=2, num_routing=5)
    #model = CnnNet(input_shape=[row,col,1], n_class=2)
    from resnet import resnet_v1
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
    import os
    model = resnet_v1(input_shape=[row,col,1],depth=20, num_classes=2)
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
    
    #scores = model.evaluate(test_X, y_test, verbose=1)
    #print('Test loss:', scores[0])
    #print('Test accuracy:', scores[1])
    '''
    y_p = randomDeepNets(model, ((train_X, y_train), (test_X, y_test)), 
                            nNet=50, seed=12345, batch_size=100)
    y_pred = (y_p>0.5).astype(float)
    
    y_pred = trainAndTest(model, ((train_X, y_train), (test_X, y_test)),
                     lr=0.001, lam_recon=0.35, batch_size=100, epochs=10)
    '''
    
    print('Test Accuracy:', accuracy_score(test_y, y_pred))
    print('Test mattews-corrcoef', matthews_corrcoef(test_y, y_pred))
    print('Test confusion-matrix', confusion_matrix(test_y, y_pred))
    


                
                
                
                
                
                
                