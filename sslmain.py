# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:37:39 2019

@author: lwzjc
"""

from configparser import ConfigParser
from semisupLearner_keras import displayMetrics
from semisupLearner_keras import SemisupLearner
from nets import semiSL2Dnet
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import KFold
import numpy as np
import os


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
    model = semiSL2Dnet((2*confParam['windown_size']+1, confParam['width'], confParam['channels'],), confParam['num_classes']) 
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

def load_data():
    from formulate import protsFormulateByXiaoInfoCode, protsFormulateByOneHotCode, protsFormulateByPhychemCode
    data = np.load('KfBenchmarkDataset.npz')
    train_posseq_ls, train_negseq_ls = data['trainPos'], data['trainNeg']
    test_posseq_ls, test_negseq_ls = data['testPos'], data['testNeg']
    
    x_train_pos, x_train_neg = [], []
    x_test_pos, x_test_neg = [], []
    for k in range(5):
        xix = protsFormulateByXiaoInfoCode(train_posseq_ls[k])
        ohx = protsFormulateByOneHotCode(train_posseq_ls[k])
        pcx = protsFormulateByPhychemCode(train_posseq_ls[k])
        # 连结三种编码，每个氨基酸编码为34维向量，最后一列是标识是否为填充列
        x_train_pos.append( np.concatenate((xix[:-1],ohx[:-1],pcx)))
            
        xix = protsFormulateByXiaoInfoCode(train_negseq_ls[k])
        ohx = protsFormulateByOneHotCode(train_negseq_ls[k])
        pcx = protsFormulateByPhychemCode(train_negseq_ls[k])
        x_train_neg.append( np.concatenate((xix[:-1],ohx[:-1],pcx)))
        
        xix = protsFormulateByXiaoInfoCode(test_posseq_ls[k])
        ohx = protsFormulateByOneHotCode(test_posseq_ls[k])
        pcx = protsFormulateByPhychemCode(test_posseq_ls[k])
        x_test_pos.append( np.concatenate((xix[:-1],ohx[:-1],pcx)))
        
        xix = protsFormulateByXiaoInfoCode(test_negseq_ls[k])
        ohx = protsFormulateByOneHotCode(test_negseq_ls[k])
        pcx = protsFormulateByPhychemCode(test_negseq_ls[k])
        x_test_pos.append( np.concatenate((xix[:-1],ohx[:-1],pcx)))
def ssl2Dpredictor():
    confParam = readConfParam()
    num_classes = confParam['num_classes']
    ws = confParam['windown_size']
    width = confParam['width']
    channels = confParam['channels']
    (X_train_pos_ls, X_test_pos_ls), (X_train_neg_ls, X_test_neg_ls) = splitDataKF(ws=confParam['windown_size'])
    y_pred, y_targ = np.zeros((0,2)), np.zeros((0,2))
    for i in range(5):
        # formulate protein seqs
        x_train_pos = protsFormulateByXiaoInfoCode(X_train_pos_ls[i])
        x_test_pos = protsFormulateByXiaoInfoCode(X_test_pos_ls[i])
        x_train_neg = protsFormulateByXiaoInfoCode(X_train_neg_ls[i])
        x_test_neg = protsFormulateByXiaoInfoCode(X_test_neg_ls[i])
        
        # bulid testing samples set and their labels
        x_test = np.concatenate((x_test_pos, x_test_neg))
        x_test = x_test.reshape(x_test.shape[0],2*ws+1,width,channels)
        
        y_test = np.zeros((len(x_test), num_classes))
        y_test[:len(x_test_pos), 1] = 1 #正样本标记
        y_test[len(x_test_pos):,0] = 1 # 负样本标记
        
        # label positive samples
        y_train_pos = np.zeros((len(x_train_pos), num_classes))
        y_train_pos[:,1] = 1
        
        # build training samples set and their labels 
        num_train_pos = x_train_pos.shape[0]
        y_train_neg = np.zeros((len(x_train_neg),num_classes))
        x_train_neg, y_train_neg = shuffle(x_train_neg, y_train_neg)
        y_train_neg[:num_train_pos,0] = 1
        
        x_train = np.concatenate((x_train_pos, x_train_neg))
        y_train = np.concatenate((y_train_pos, y_train_neg))
        x_train = x_train.reshape(x_train.shape[0],2*ws+1,width,channels)
        
        model_name = 'keras_pdna224_trained_5fold_model_{}.h5'.format(i)
        save_dir = os.path.join(os.getcwd(), confParam['save_dir'])
        modelFile = os.path.join(save_dir, model_name)
        noteInfo = '\nOn bechmark dataset, semi-supervised learning predicting result'
        metricsFile = 'semisup_info.txt'

        pred = semisupLearn(x_train, y_train, x_test, y_test, modelFile, noteInfo, metricsFile, **confParam)
                        
        y_t = np.argmax(y_test,-1)
        y_p = np.argmax(pred,-1)
        # metrics the predictor
        print('acc={}'.format(accuracy_score(y_t, y_p)))  
        print('mcc = {}'.format(matthews_corrcoef(y_t, y_p)))
        
        y_pred = np.concatenate((y_pred, pred))
        y_targ = np.concatenate((y_targ, y_test))  

    # 5-kf metrics
    y_t = np.argmax(y_targ,-1)
    y_p = np.argmax(y_pred,-1)
    
    print('acc={}'.format(accuracy_score(y_t, y_p)))  
    print('mcc = {}'.format(matthews_corrcoef(y_t, y_p)))

if __name__=="__main__":
    ssl2Dpredictor()



