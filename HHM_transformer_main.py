# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:44:55 2020

@author: lwzjc
"""


from Bio import SeqIO
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tools import displayMetrics, displayMLMetrics, plot_history

def loadHHM():
    from imblearn.over_sampling import ADASYN
    traindata = np.load('PDNA543_HHM_7.npz', allow_pickle=True)
    train_pos, train_neg = traindata['pos'], traindata['neg']
    testdata = np.load('PDNA543TEST_HHM_7.npz', allow_pickle=True)
    test_pos, test_neg = testdata['pos'], testdata['neg']
    
    x_train = np.concatenate((train_pos, train_neg))
    y_train = [1 for _ in range(train_pos.shape[0])] + [0 for _ in range(train_neg.shape[0])]
    #y_train = keras.utils.to_categorical(y_train, num_classes=2)
    x_train = x_train.reshape(-1, 450)
    ada = ADASYN(random_state=42)
    x_train_res, y_train_res = ada.fit_resample(x_train, y_train)
    x_train_res = x_train_res.reshape((-1, 15,30))
    y_train_res = keras.utils.to_categorical(y_train_res, num_classes=2)
    
    x_test = np.concatenate((test_pos, test_neg))
    y_test = [1 for _ in range(test_pos.shape[0])] + [0 for _ in range(test_neg.shape[0])]
    y_test = keras.utils.to_categorical(y_test, num_classes=2)
    
    return (x_train_res, y_train_res), (x_test, y_test)

"""
位置函数
    基于角度的位置编码方法。计算位置编码矢量的长度
    Parameters
    ----------
    pos : 
        在句子中字的位置序号，取值范围是[0, max_sequence_len).
    i   : int
        字向量的维度，取值范围是[0, embedding_dim).
    embedding_dim : int
        字向量最大维度， 即embedding_dim的最大值.

    Returns
    -------
    float32
        第pos位置上对应矢量的长度.
"""
def get_angles(pos, i, embed_dim):
    angel_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
    return abs(pos) * angel_rates

# 位置编码
def position_encoding(position, embed_dim):
    angel_rads = get_angles(np.arange(-position, position+1)[:, np.newaxis], 
                            np.arange(embed_dim)[np.newaxis, :], 
                            embed_dim)
    pos_encoding = np.zeros(angel_rads.shape)
    for i in range(embed_dim):
        if i%2 == 0:
            pos_encoding[:,i] = np.sin(angel_rads[:,i])
        else:
            pos_encoding[:,i] = np.cos(angel_rads[:,i])
    #pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)



# 自注意力机制
def scaled_dot_product_attention(q, k, v, mask=None):
    
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # 掩码.将被掩码的token乘以-1e9（表示负无穷），这样
    # softmax之后就为0， 不对其它token产生影响
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # attention乘上value
    output = tf.matmul(attention_weights, v) # (..., seq_len_v, depth)
    
    return output, attention_weights

class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # embd_dim必须可以被num_heads整除
        assert embed_dim % num_heads == 0
        # 分头后的维度
        self.projection_dim = embed_dim // num_heads
        self.wq = layers.Dense(embed_dim)
        self.wk = layers.Dense(embed_dim)
        self.wv = layers.Dense(embed_dim)
        self.dense = layers.Dense(embed_dim)
        
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        # 分头前的前向网络，获取q, k, v语义
        q = self.wq(q)
        k = self.wq(k)
        v = self.wv(v)
        
        # 分头
        q = self.separate_heads(q, batch_size) # [batch_size, num_heads, seq_len_q, projection_dim]
        k = self.separate_heads(k, batch_size)
        v = self.separate_heads(v, batch_size)
        
        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        # 合并多头
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embed_dim))
        
        # 全连接重塑
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
# 构造前向网络
def point_wise_feed_forward_network(d_model, diff):
    # d_model 即embed_dim
    return tf.keras.Sequential([
        layers.Dense(diff, activation='relu'),
        layers.Dense(d_model)])    
    
# transformer编码层
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, n_heads, ffd, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, ffd)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training, mask=None):
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs + att_output)
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class Encoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ffd,
                 max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()
        
        self.n_layers = n_layers
        self.d_model = d_model
        self.pos_embedding = position_encoding(max_seq_len//2, d_model)
        self.encoder_layer = [EncoderLayer(d_model, n_heads, ffd, dropout_rate)
                              for _ in range(n_layers)]
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training, mask=None):
        word_emb = tf.cast(inputs, tf.float32)
        #word_emb *= (tf.cast(self.d_model, tf.float32))
        emb = word_emb + self.pos_embedding
        
        x = self.dropout(emb, training=training)
        for i in range(self.n_layers):
            x = self.encoder_layer[i](x, training, mask)
        return x
    
def buildModel(maxlen, embed_dim, num_heads, ff_dim, 
               num_blocks, droprate, fl_size, num_classes):
    inputs = layers.Input(shape=(maxlen,30))
    
    encoder = Encoder(n_layers=num_blocks, d_model=embed_dim, n_heads=num_heads, 
                      ffd=ff_dim, max_seq_len=maxlen, dropout_rate=droprate)
    x = encoder(inputs, True)
    
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(droprate)(x)
    x = layers.Dense(fl_size, activation="relu")(x)
    x = layers.Dropout(droprate)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model    

def transformer_predictor(X_train, y_train, X_test, y_test, modelfile, params):
    keras.backend.clear_session()

    model = buildModel(params['maxlen'], params['embed_dim'], 
                    params['num_heads'], params['ff_dim'],  params['num_blocks'], 
                    params['droprate'], params['fl_size'], params['num_classes'])
    model.summary()

    checkpoint = callbacks.ModelCheckpoint(modelfile, monitor='val_loss',
                                       save_best_only=True, 
                                       save_weights_only=True, 
                                       verbose=1)
    history = model.fit(
        X_train, y_train, 
        batch_size=params['batch_size'], epochs=params['epochs'], 
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
        )

    plot_history(history)

    #model.load_weights(modelfile)
    score = model.predict(X_test)
    
    return score

# transformer net params
params = {}
params['maxlen'] = 15
params['embed_dim'] = 30 # Embedding size for each token
params['num_heads'] = 6  # Number of attention heads
params['ff_dim'] = 128  # Hidden layer size in feed forward network inside transformer
params['num_blocks'] = 12
params['droprate'] = 0.2
params['fl_size'] = 128
params['num_classes'] = 2
params['epochs'] = 20
params['batch_size'] = 200


# load data
(x_train, y_train),(x_test, y_test) = loadHHM()

x_train, y_train = shuffle(x_train, y_train)
# training and test
modelfile = './save_models/hhm_trainsformer.h5'
score = transformer_predictor(x_train, y_train, x_test, y_test, modelfile, params)
pred = np.argmax(score, 1)
displayMetrics(np.argmax(y_test, 1), pred)


