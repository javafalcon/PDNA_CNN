# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:42:03 2020

@author: lwzjc
"""
import numpy as np
from keras import layers, models, optimizers,losses
from keras import backend as K
from keras.models import Model
import tensorflow as tf
#from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask, squash
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score,recall_score,precision_score
from sklearn.utils import class_weight, shuffle, resample
import os
from keras import callbacks
import argparse
from SemisupCallback import SemisupCallback

def CapsNet(input_shape, n_class, num_routing, 
            conv_filters=128,kernel_size=7, padding='valid',
            prim_cap_dim=16, dig_cap_dim=32):
    """
    A Capsule Network on PDNA-543.
    :param input_shape: data shape, 3d, [width, height, channels],for example:[21,28,1]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=conv_filters, kernel_size=kernel_size, strides=1, padding=padding, activation='relu', name='conv1')(x)
    drop1 = layers.Dropout(0.25)
    
    conv1_1 = drop1(conv1)
    conv1_2 = drop1(conv1)
    
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    #primarycaps = PrimaryCap(conv1_1, dim_vector=prim_cap_dim, n_channels=dig_cap_dim, kernel_size=kernel_size, strides=2, padding=padding)
    primarycaps = layers.Conv2D(filters=prim_cap_dim*dig_cap_dim, kernel_size=kernel_size, strides=2, padding=padding,
                           name='primarycap_conv2d')
    primarycaps_1 = primarycaps(conv1_1)
    primarycaps_2 = primarycaps(conv1_2)
    
    drop2 = layers.Dropout(0.25)
    primarycaps_1 = drop2(primarycaps_1)
    primarycaps_2 = drop2(primarycaps_2)
    
    primarycaps_1 = layers.Reshape(target_shape=[-1, prim_cap_dim], name='primarycap_reshape_1')(primarycaps_1)
    primarycaps_2 = layers.Reshape(target_shape=[-1, prim_cap_dim], name='primarycap_reshape_2')(primarycaps_2)
    
    primarySquash = layers.Lambda(squash, name='primarycap_squash')
    primarySquash_1 = primarySquash(primarycaps_1)
    primarySquash_2 = primarySquash(primarycaps_2)
     
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=dig_cap_dim, num_routing=num_routing, name='digitcaps')
    digitcaps_1 = digitcaps(primarySquash_1)
    digitcaps_2 = digitcaps(primarySquash_2)
    
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')
    out_caps_1 = out_caps(digitcaps_1)
    out_caps_2 = out_caps(digitcaps_2)
    
    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps_1, y])  # The true label is used to mask the output of capsule layer.
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(np.prod(input_shape), activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps_1, out_caps_2, x_recon])
    """
    return models.Model(inputs=x, outputs=out_caps)
    """
def mcc(y_true, y_pred):
    return matthews_corrcoef(K.eval(K.argmax(y_true,1)), K.eval(K.argmax(y_pred,1)))

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    m = K.sum(y_true, axis=-1)
    return  K.switch(K.equal(K.sum(y_true), 0), 0., K.sum(K.categorical_crossentropy(tf.boolean_mask(y_true,m), tf.boolean_mask(y_pred,m), from_logits=True)) / K.sum(y_true))

    #L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
    #    0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    #return K.mean(K.sum(L, 1))

def unsup_loss(o1, o2):
    def los(y_true, y_pred):
        return losses.mean_squared_error(o1, o2)
    return los

def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    weight = K.variable(0.)
    ssCallback = SemisupCallback(weight,rampup_length=50, rampdown_length=30, 
                                     epochs=args.epochs, learning_rate_max=0.03, 
                                     scaled_unsup_weight_max=100, gammer=5.0, 
                                     beita=0.5)
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.weights,
                                           save_best_only=True, save_weights_only=True, verbose=0)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))
    
    capslayer = model.get_layer('out_caps')
    unsuploss = unsup_loss(capslayer.get_output_at(0), capslayer.get_output_at(1))
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, unsuploss, 'mse'],
                  loss_weights=[1., weight, args.lam_recon],
                  metrics={'out_caps': "accuracy"})
    
    # Training without data augmentation:
    if len(x_test) == 0:
        model.fit([x_train, y_train], [y_train, y_train, x_train], 
              batch_size=args.batch_size, 
              epochs=args.epochs,
              validation_split=0.1, 
              callbacks=[ssCallback,log, tb, checkpoint, lr_decay])
    else:    
        model.fit([x_train, y_train], [y_train, y_train, x_train], 
              batch_size=args.batch_size, 
              epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, y_test, x_test]], 
              callbacks=[log, tb, checkpoint, lr_decay])
    
    #model.save_weights(args.save_dir + '/trained_model.h5')
    #print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, batch_size=100):
    x_test, y_test = data
    y_pred, y_, x_recon = model.predict([x_test, y_test], batch_size=batch_size)
    print('-'*50)
    y_p = np.argmax(y_pred, 1)
    y_t = np.argmax(y_test,1)
    print('Test Accuracy:', accuracy_score(y_t, y_p))
    print('Test mattews-corrcoef', matthews_corrcoef(y_t, y_p))
    return y_pred


def build_test(x_test_pos, x_test_neg):
    x_test = np.concatenate((x_test_pos, x_test_neg))
    y_test = np.zeros((x_test.shape[0],2))
    y_test[:x_test_pos.shape[0], 1] = 1
    y_test[x_test_pos.shape[0]:,0] = 1
    x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1).astype('float32')
    
    return  (x_test, y_test)

def build_resampleTrain(x_train_pos, x_train_neg, neg_samples=1):
    x_neg = resample(x_train_neg, n_samples=x_train_pos.shape[0]*neg_samples, replace=False)
    x_train = np.concatenate((x_train_pos, x_neg))
    y_train = np.zeros((x_train.shape[0],2))
    y_train[:x_train_pos.shape[0],1] = 1
    y_train[x_train_pos.shape[0]:,0] = 1
    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1).astype('float32')
    x_train, y_train = shuffle(x_train, y_train)
    
    return (x_train, y_train)

def build_semisupTrain(x_train_pos, x_train_neg, rate=0.618):
    n_pos_samples = x_train_pos.shape[0]
    n_samples = int(n_pos_samples * rate)
    x_train_neg = shuffle(x_train_neg)
    x_train_pos = shuffle(x_train_pos)
    x_train = np.concatenate((x_train_pos, x_train_neg))
    y_train = np.zeros((x_train.shape[0],2))
    y_train[:n_samples,1] = 1
    y_train[n_pos_samples:n_pos_samples+n_samples,0] = 1
    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1).astype('float32')
    x_train, y_train = shuffle(x_train, y_train)
    
    return (x_train, y_train)

def writeMetrics(metricsFile, y_true, predicted_Probability, noteInfo=''):
    #predicts = np.array(predicted_Probability> 0.5).astype(int)
    #predicts = predicts[:,0]
    #labels = y_true[:,0]
    predicts = np.argmax(predicted_Probability, axis=1)
    labels = np.argmax(y_true, axis=1)
    cm=confusion_matrix(labels,predicts)
    with open(metricsFile,'a') as fw:
        if noteInfo:
            fw.write(noteInfo + '\n')
        for i in range(2):
            fw.write(str(cm[i,0]) + "\t" +  str(cm[i,1]) + "\n" )
        fw.write("ACC: %f "%accuracy_score(labels,predicts))
        fw.write("\nF1: %f "%f1_score(labels,predicts))
        fw.write("\nRecall: %f "%recall_score(labels,predicts))
        fw.write("\nPre: %f "%precision_score(labels,predicts))
        fw.write("\nMCC: %f "%matthews_corrcoef(labels,predicts))
        fw.write("\nAUC: %f\n "%roc_auc_score(labels,predicted_Probability[:,0]))

def trainMain():
    #import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lam_recon', default=0.345, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=5, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result/PDNA-543')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default='./result/PDNA-224/trained_model.h5')
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # load data
    #(x_train, y_train), (x_test, y_test) = load_PDNA543_hhm()
    traindatafile = 'PDNA224_HHM_7.npz'
    
    from dataset224 import load_kf_data
    
    (kf_x_pos_train, kf_x_neg_train), (kf_x_pos_test, kf_x_neg_test)  = load_kf_data(benckmarkFile=traindatafile,k=10)
    y_ps, y_ts = np.zeros((0,2)), np.zeros((0,2))
    N = 2
    for i in range(1):
        (x_test, y_test) = build_test(kf_x_pos_test[i], kf_x_neg_test[i])
        
        (x_train, y_train) = build_semisupTrain(kf_x_pos_train[i], kf_x_neg_train[i])
        
        y_pred = np.zeros(shape=(y_test.shape[0],2))
        
        # define model
        model = CapsNet(input_shape=x_train.shape[1:],
                        n_class=len(np.unique(np.argmax(y_train, 1))),
                        num_routing=args.num_routing, kernel_size=7, 
                        padding='same', prim_cap_dim=8, 
                        dig_cap_dim=16, conv_filters=32
                        )
        model.summary()
        #plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)
        
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
        
        y_p = test(model=model, data=(x_test, y_test))
        
        K.clear_session()
        tf.reset_default_graph()
  
        y_ps = np.concatenate((y_ps, y_p))
        y_ts = np.concatenate((y_ts, y_test))
    writeMetrics('PDNA224_result.txt', y_ts, y_ps, 'Total Predicted Metrics:'.format(i))
    
def GANTrain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lam_recon', default=0.225, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=5, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result/PDNA-224')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default='./result/PDNA-224/trained_model.h5')
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # load data
    traindatafile = 'PDNA224_OneHot_7.npz'
    data = np.load(traindatafile)
    x_pos, x_neg = data['pos'], data['neg']
    x_pos = shuffle(x_pos)
    x_neg = shuffle(x_neg)
    
    x_test = np.concatenate((x_pos[:100], x_neg[:100]))
    y_test = np.zeros((len(x_test), 2))
    y_test[:100, 0] = 1
    y_test[100:, 1] = 1
    x_test = np.reshape(x_test, (-1, 15, 20, 1))
    
    x_pos_2, x_neg_2 = x_pos[100:], x_neg[100:]
    x_n = resample(x_neg_2, n_samples=x_pos_2.shape[0]*2, replace=False)
    
    x_train = np.concatenate((x_pos_2, x_n))
    y_train = np.zeros((len(x_train), 2))
    num_pos = len(x_pos_2)
    y_train[num_pos:2*num_pos, 1] = 1
    y_train[:num_pos, 0] = 1
    x_train = np.reshape(x_train, (-1, 15, 20, 1))
    
    
    model = CapsNet(input_shape=x_train.shape[1:],
                            n_class=2,
                            num_routing=args.num_routing, kernel_size=7, 
                            padding='same', prim_cap_dim=8, 
                            dig_cap_dim=16, conv_filters=64
                            )
    model.summary()
                 
    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
 
    model.load_weights(args.weights)      
    
    gen_model = Model(input=model.input, output=model.get_layer('digitcaps').output)
    gen_model_output = gen_model.predict([x_train, y_train])
    
    X = np.reshape(gen_model_output, (-1, 32))
    return X, y_train

if __name__ == '__main__':
    
    X, y = trainMain() 
    """
    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.is_training:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=model, data=(x_test, y_test))
   """