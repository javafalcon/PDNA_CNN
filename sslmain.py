# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:37:39 2019

@author: lwzjc
"""

from configparser import ConfigParser
from semisupLearner_keras import displayMetrics
from semisupLearner_keras import SemisupLearner
from nets import semiSL2Dnet,supLearnNet
from resnet_keras import resnet_v1
from sklearn.utils import shuffle, resample
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import KFold
import numpy as np
import os
from keras import optimizers
from keras import backend as K
from keras.utils import to_categorical
from util import pcaval
from imblearn.over_sampling import ADASYN

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

def load_upsample_TrainData(df, row, col, seed):
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

def load_updownsample_TrainData(df, row, col, alpa=3, seed=None):
    data = np.load(df)#'PDNA_Data\\PDNA_543_train_7.npz'
    train_pos_seqs = data['pos'] 
    train_neg_seqs = data['neg']
    train_pos_X = pcaval(train_pos_seqs)
    train_neg_X = pcaval(train_neg_seqs)
    train_neg_X = shuffle(train_neg_X)
    
    X_train = np.concatenate((train_pos_X, train_neg_X[:int(len(train_pos_X)*alpa)]))
    y_train = np.zeros((len(X_train),))
    y_train[:len(train_pos_X)] = 1.

    ada = ADASYN(random_state=seed)
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

def readConfParam():
    conf = ConfigParser()
    conf.read('conf.ini')
    confParam = {}
    
    confParam['windown_size'] = int(conf.get('seqparam','windown_size'))
    confParam['width'] = int(conf.get('seqparam','width'))
    confParam['channels'] = int(conf.get('seqparam','channels'))
    confParam['num_classes'] = int(conf.get('netparam', 'num_classes'))
    confParam['batch_size'] = int(conf.get('netparam', 'batch_size'))
    confParam['epochs'] = int(conf.get('netparam', 'epochs'))
    confParam['patience'] = int(conf.get('netparam', 'patience'))
    confParam['learning_rate'] = float(conf.get('netparam', 'learning_rate'))
    confParam['save_dir'] = conf.get('netparam', 'save_dir')
    
    confParam['rampup_length'] = int(conf.get('semisupparam', 'rampup_length'))
    confParam['rampdown_length'] = int(conf.get('semisupparam', 'rampdown_length'))
    confParam['learning_rate_max'] = float(conf.get('semisupparam', 'learning_rate_max'))
    confParam['scaled_unsup_weight_max'] = int(conf.get('semisupparam', 'scaled_unsup_weight_max'))
    confParam['gammer'] = float(conf.get('semisupparam', 'gammer'))
    confParam['beita'] = float(conf.get('semisupparam', 'beita'))
     
    return confParam

def semisupLearn(x_train, y_train, x_test, y_test, modelFile, noteInfo, metricsFile, **confParam):
    # semi-supervised learning
    model = semiSL2Dnet((confParam['windown_size'], confParam['width'], confParam['channels'],), confParam['num_classes']) 
    #model = resnet_v1(input_shape=[confParam['windown_size'], confParam['width'], confParam['channels']], depth=20, num_classes=confParam['num_classes'])
    ssparam={}
    ssparam['x_train'] = x_train
    ssparam['y_train'] = [y_train, y_train]
    ssparam['save_dir'] = confParam['save_dir']
    ssparam['batch_size'] = confParam['batch_size']
    ssparam['epochs'] = confParam['epochs']
    ssparam['patience'] = confParam['patience'] 
    ssparam['rampup_length'] = confParam['rampup_length']
    ssparam['rampdown_length'] = confParam['rampdown_length']
    ssparam['learning_rate_max'] = confParam['learning_rate_max']
    ssparam['scaled_unsup_weight_max'] = confParam['scaled_unsup_weight_max']
    ssparam['gammer'] = confParam['gammer']
    ssparam['beita'] = confParam['beita'],
    ssparam['learning_rate'] = confParam['learning_rate']
    ssl = SemisupLearner(modelFile, model, **ssparam)
    # Train net
    ssl.train()
    # predict
    pred_prob = ssl.predict(x_test)
    # print predicting metrics
    displayMetrics(y_test, pred_prob, noteInfo, metricsFile) 

    return pred_prob


def supLearn(x_train, y_train, x_test, y_test, modelFile, noteInfo, metricsFile, **confParam):
    # semi-supervised learning
    K.clear_session()
    model = supLearnNet((2*confParam['windown_size']+1, confParam['width'], confParam['channels'],), confParam['num_classes']) 
    #save_dir = confParam['save_dir']
    batch_size = confParam['batch_size']
    epochs = confParam['epochs']
    learning_rate = confParam['learning_rate']
    #patience = confParam['patience']
    model.compile(loss= 'categorical_crossentropy', 
                     optimizer=optimizers.Adam(lr=learning_rate),  
                     metrics=['accuracy']) 
    model.fit(x_train,y_train,
              batch_size=batch_size,
              epochs=epochs, 
              validation_data=resample(x_test,y_test,n_samples=4000)
              )
    
    # predict
    pred_prob = model.predict(x_test)
    #K.clear_session()
    #K.tf.reset_default_graph()
    
    # print predicting metrics
    displayMetrics(y_test, pred_prob, noteInfo, metricsFile) 

    return pred_prob


# 集成。M:神经网络个数, r:样本抽样比例, f:特征个数     
def ensmbSSL2Dpredictor(M, alpa, trainFile, testFile):
    confParam = readConfParam()
    num_classes = confParam['num_classes']
    row = confParam['windown_size']
    col = confParam['width']
    channels = confParam['channels']
    
    (X_test, test_y) = load_TestData(testFile)
    X_test = X_test.reshape(-1,row,col,1)
    y_test = to_categorical(test_y, num_classes=2)
     

    pred = np.zeros((len(y_test),2))
    
    random_state = [42,163,1436,57,342,93,3987,1135,198743]
    for i in range(M):
        (X_train, train_y) = load_updownsample_TrainData(trainFile, row, col, alpa, random_state[i]) 
        X_train = X_train.reshape(-1,row,col,channels)
        y_train = to_categorical(train_y, num_classes=num_classes)
        
        noteInfo = "Ensemble {}/{} semisuper learners on downsample dataset \
                    (Numbers of negative = {} * Numbers of postive".format(i, M, alpa)            
        modelFile = "save_models/ensem_semisup_downsample_{}.hdf5".format(i)
        
        metricsFile = 'semisup_info_2.txt'
    
        p = semisupLearn(X_train, y_train, X_test, y_test, modelFile, noteInfo, metricsFile, **confParam)
            
        p = (p > 0.5).astype(int)
        pred = pred+p
           
        
    pred = pred/M        
    y_t = np.argmax(y_test,-1)
    y_p = np.argmax(pred,-1)
    # metrics the predictor
    print('acc={}'.format(accuracy_score(y_t, y_p)))  
    print('mcc = {}'.format(matthews_corrcoef(y_t, y_p)))
            

if __name__=="__main__":
    ensmbSSL2Dpredictor(9, alpa=3, testFile='PDNA_543_test_7.npz', trainFile='PDNA_543_train_7.npz')




