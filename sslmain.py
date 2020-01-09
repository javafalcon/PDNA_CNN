# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:37:39 2019

@author: lwzjc
"""

from configparser import ConfigParser
from semisupLearner_keras import displayMetrics
from semisupLearner_keras import SemisupLearner
from nets import semiSL2Dnet,supLearnNet
from sklearn.utils import shuffle, resample
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import KFold
import numpy as np
import os
from keras import optimizers
from keras import backend as K
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

def load_Kfdata(benchmarkDataFile,k):
    from formulate import protsFormulateByXiaoInfoCode, protsFormulateByOneHotCode, protsFormulateByPhychemCode
    data = np.load(benchmarkDataFile, allow_pickle='True')
    train_posseq_ls, train_negseq_ls = data['trainPos'], data['trainNeg']
    test_posseq_ls, test_negseq_ls = data['testPos'], data['testNeg']
       
    xix = protsFormulateByXiaoInfoCode(train_posseq_ls[k])
    ohx = protsFormulateByOneHotCode(train_posseq_ls[k])
    pcx = protsFormulateByPhychemCode(train_posseq_ls[k])
    # 连结三种编码，每个氨基酸编码为34维向量，最后一列是标识是否为填充列
    x_train_pos = np.block([xix[:,:,:-1],ohx[:,:,:-1],pcx])
        
    xix = protsFormulateByXiaoInfoCode(train_negseq_ls[k])
    ohx = protsFormulateByOneHotCode(train_negseq_ls[k])
    pcx = protsFormulateByPhychemCode(train_negseq_ls[k])
    x_train_neg = np.block([xix[:,:,:-1],ohx[:,:,:-1],pcx])
    
    xix = protsFormulateByXiaoInfoCode(test_posseq_ls[k])
    ohx = protsFormulateByOneHotCode(test_posseq_ls[k])
    pcx = protsFormulateByPhychemCode(test_posseq_ls[k])
    x_test_pos = np.block([xix[:,:,:-1],ohx[:,:,:-1],pcx])
    
    xix = protsFormulateByXiaoInfoCode(test_negseq_ls[k])
    ohx = protsFormulateByOneHotCode(test_negseq_ls[k])
    pcx = protsFormulateByPhychemCode(test_negseq_ls[k])
    x_test_neg = np.block([xix[:,:,:-1],ohx[:,:,:-1],pcx])
    
    return x_train_pos, x_train_neg, x_test_pos, x_test_neg
    
# 集成。M:神经网络个数, r:样本抽样比例, f:特征个数     
def ensmbSSL2Dpredictor(KfBenchmarkDataFile, M, rate_samples, num_features):
    confParam = readConfParam()
    num_classes = confParam['num_classes']
    ws = confParam['windown_size']
    width = confParam['width']
    channels = confParam['channels']
       
    y_pred, y_targ = np.zeros((0,2)), np.zeros((0,2))
    for i in range(5):
        x_train_pos, x_train_neg, x_test_pos, x_test_neg = load_Kfdata(KfBenchmarkDataFile, i)       
        # bulid testing samples set and their labels
        x_test = np.concatenate((x_test_pos, x_test_neg))
        
        y_test = np.zeros((len(x_test), num_classes))
        y_test[:len(x_test_pos), 1] = 1 #正样本标记
        y_test[len(x_test_pos):,0] = 1 # 负样本标记
        
        # label positive samples
        y_train_pos = np.zeros((len(x_train_pos), num_classes))
        y_train_pos[:,1] = 1
        
        # build training samples set and their labels 
        y_train_neg = np.zeros((len(x_train_neg),num_classes))
        y_train_neg[:,0] = 1
        
        # 生成9个随机特征之空间和随机样本空间，得到训练集，然后集成
        num_samples = int(x_train_pos.shape[0] * rate_samples)
        features_indx = list(range(x_train_pos.shape[2]))
        pred = np.zeros((len(y_test),2))
        for m in range(M):
            # 随机取num_train_pos个正样本
            x_p, y_p = resample(x_train_pos, y_train_pos, n_samples=num_samples, replace=False)
            
            # 随机抽取2*num_train_pos个负样本，其中把一半的样本去标签
            x_n, y_n = resample(x_train_neg, y_train_neg, n_samples=2*num_samples, replace=False)
            y_n[num_samples:,0] = 0
            
            fid = resample(features_indx, n_samples=num_features, replace=False)
            x = np.concatenate((x_p[:,:,fid], x_n[:,:,fid]))
            y = np.concatenate((y_p, y_n))
            x = x.reshape(x.shape[0],x.shape[1],x.shape[2],channels)
            x_t = x_test[:,:,fid]
            x_t = x_t.reshape(x_t.shape[0], x_t.shape[1], x_t.shape[2],channels)
            
            model_name = 'keras_pdna224_trained_5fold_model_{}.h5'.format(i)
            save_dir = os.path.join(os.getcwd(), confParam['save_dir'])
            modelFile = os.path.join(save_dir, model_name)
            noteInfo = '\nOn bechmark dataset, semi-supervised learning predicting result'
            metricsFile = 'semisup_info.txt'
    
            p = semisupLearn(x, y, x_t, y_test, modelFile, noteInfo, metricsFile, **confParam)
            #p = supLearn(x, y, x_t, y_test, modelFile, noteInfo, metricsFile, **confParam)
            pred = pred+p
        
        pred = pred/M        
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
    ensmbSSL2Dpredictor('KfBenchmarkDataset_20.npz',10,0.7,24)




