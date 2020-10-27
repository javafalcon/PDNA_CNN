# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:55:24 2020

@author: lwzjc
"""
from configparser import ConfigParser
from collections import Counter
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tools import displayMetrics, plot_history, plot_cm
from transformer_v3 import create_padding_mask, Encoder
from semisupLearner_keras import displayMetrics
from semisupLearner_keras import sup_loss, SemisupLearner

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def readConfParam(confName):
    conf = ConfigParser()
    conf.read(confName)
    confParam = {}
    
    confParam['maxlen'] = int(conf.get('transformer-param', 'maxlen'))
    confParam['embed_dim'] = int(conf.get('transformer-param', 'embed_dim'))
    confParam['num_heads'] = int(conf.get('transformer-param', 'num_heads'))
    confParam['ff_dim'] = int(conf.get('transformer-param', 'ff_dim'))
    confParam['num_blocks'] = int(conf.get('transformer-param', 'num_blocks'))
    confParam['droprate'] = float(conf.get('transformer-param', 'droprate'))
    confParam['fc_size'] = int(conf.get('transformer-param', 'fc_size'))
    confParam['num_classes'] = int(conf.get('transformer-param', 'num_classes'))
    
    confParam['batch_size'] = int(conf.get('training-param', 'batch_size'))
    confParam['epochs'] = int(conf.get('training-param', 'epochs'))
    confParam['patience'] = int(conf.get('training-param', 'patience'))
    confParam['learning_rate'] = float(conf.get('training-param', 'learning_rate'))
    
    confParam['rampup_length'] = int(conf.get('semisup-param', 'rampup_length'))
    confParam['rampdown_length'] = int(conf.get('semisup-param', 'rampdown_length'))
    confParam['learning_rate_max'] = float(conf.get('semisup-param', 'learning_rate_max'))
    confParam['scaled_unsup_weight_max'] = int(conf.get('semisup-param', 'scaled_unsup_weight_max'))
    confParam['gammer'] = float(conf.get('semisup-param', 'gammer'))
    confParam['beita'] = float(conf.get('semisup-param', 'beita'))
    
    return confParam

def loadHHM_RUS(trainfile, testfile):
    traindata = np.load(trainfile, allow_pickle=True)
    train_pos, train_neg = traindata['pos'], traindata['neg']
    testdata = np.load(testfile, allow_pickle=True)
    test_pos, test_neg = testdata['pos'], testdata['neg']
    
    x_test = np.concatenate((test_pos, test_neg))
    y_test = [1 for _ in range(test_pos.shape[0])] + [0 for _ in range(test_neg.shape[0])]
    y_test = np.array(y_test, dtype=float)
    
    # 计算正类样本和负类样本数量：pos-正样本数量；neg-负样本数量
    pos, neg = train_pos.shape[0], train_neg.shape[0]
    k = neg // (2*pos)
    
    # RUS随机下采样，负样本数量=2*pos
    indices = np.arange(neg)
    np.random.shuffle(indices)
    x_train_ls, y_train_ls = [], []
    for i in range(k):
        start = i * pos
        end = (i+1)*pos if (i+1)*pos < neg else neg
        x_res_neg = train_neg[start:end]
        x_train = np.concatenate((train_pos, x_res_neg))
        y_train = [1] * pos + [0] * x_res_neg.shape[0]
        y_train = np.array(y_train, dtype=float)
        x_train_ls.append(x_train)
        y_train_ls.append(y_train)
    return (x_train_ls, y_train_ls), (x_test, y_test)
        

    
def buildModel(maxlen, embed_dim, num_heads, ff_dim, 
               num_blocks, droprate, fc_size, 
               num_classes, metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        
    inputs = layers.Input(shape=(maxlen,30))
    mask = create_padding_mask(inputs)
    encoder = Encoder(n_layers=num_blocks, d_model=embed_dim, n_heads=num_heads, 
                      ffd=ff_dim, max_seq_len=maxlen, dropout_rate=droprate)
    x = encoder(inputs, True, mask)
    
    x = layers.GlobalMaxPooling1D()(x)
    drop_1 = layers.Dropout(droprate)
    x_a = drop_1(x)
    x_b = drop_1(x)
    
    dense = layers.Dense(fc_size, activation="relu")
    drop_2 = layers.Dropout(droprate, name="unsupLayer")
    x_a = dense(x_a)
    x_a = drop_2(x_a)
    
    x_b = dense(x_b)
    x_b = drop_2(x_b)
    
    outputs = layers.Dense(num_classes, activation="sigmoid", bias_initializer=output_bias)(x_a)
    
    model = keras.Model(inputs=inputs, outputs=[outputs, x_b])
    
    model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)
    model.summary()
    return model    

def semisupLearn(x_train, y_train, x_test, y_test, modelFile, noteInfo, metricsFile, **confParam):
    # build model
    model = buildModel(confParam['maxlen'], confParam['embed_dim'],
                       confParam['num_heads'], confParam['ff_dim'], 
                       confParam['num_blocks'], confParam['droprate'],
                       confParam['fc_size'], confParam['num_classes']) 
    
    # semi-supervised learning
    
    ssparam={}
    ssparam['x_train'] = x_train
    ssparam['y_train'] = [y_train, y_train]
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
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_auc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)
    
    ssl = SemisupLearner(modelFile, model, **ssparam, earlystopping=early_stopping)
    # Train net
    ssl.train()
    # predict
    pred_prob = ssl.predict(x_test)
    # print predicting metrics
    #displayMetrics(y_test, pred_prob, noteInfo, metricsFile)  
    return pred_prob

def ensemb_transformer_predictor(x_train_ls, y_train_ls, x_test, y_test, **confParams):    
    modelFile = './save_models/hhm_trainsformer.h5'
    noteInfo = 'HHM-transformer-semisup'
    metricsFile = 'semiup_info_2.txt'
    y_ = np.zeros((x_test.shape[0],))
    for x_train, y_train in zip(x_train_ls, y_train_ls): 
        x_train, y_train = shuffle(x_train, y_train)   
        keras.backend.clear_session()
      
        score = semisupLearn(x_train, y_train, x_test, y_test, modelFile, noteInfo, metricsFile, **confParams)

        y_ += (score[:,0]>0.5)
        
    return y_/len(x_train_ls)

# transformer net params
confParams = readConfParam('transformer-semisup-conf.ini')

(x_train_ls, y_train_ls), (x_test, y_test) = loadHHM_RUS('PDNA543_HHM_15.npz', 'PDNA543TEST_HHM_15.npz')

score = ensemb_transformer_predictor(x_train_ls, y_train_ls, x_test, y_test, **confParams)
pred = score > 0.5
for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    print("threshold=", t)
    displayMetrics(y_test, score, threshold=t)
    print("")
plot_cm(y_test, score)
