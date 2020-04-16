# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:42:03 2020

@author: lwzjc
"""
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
import tensorflow as tf
#from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score,recall_score,precision_score
from sklearn.utils import class_weight, shuffle, resample


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
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=prim_cap_dim, n_channels=dig_cap_dim, kernel_size=kernel_size, strides=2, padding=padding)

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=dig_cap_dim, num_routing=num_routing, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)
    
    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer.
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(np.prod(input_shape), activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])
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
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


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
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    # compile the model
    """
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': mcc})
    """
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss,'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': "accuracy"})
    
    # Training without data augmentation:
    
    model.fit([x_train, y_train], [y_train, x_train], 
              batch_size=args.batch_size, 
              epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], 
              callbacks=[log, tb, checkpoint, lr_decay])
    
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, batch_size=100):
    x_test, y_test = data
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=batch_size)
    print('-'*50)
    y_p = np.argmax(y_pred, 1)
    y_t = np.argmax(y_test,1)
    print('Test Accuracy:', accuracy_score(y_t, y_p))
    print('Test mattews-corrcoef', matthews_corrcoef(y_t, y_p))
    return y_pred


def load_data(k, npzfile):
    data = np.load(npzfile, allow_pickle=True)
    
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test =  data['x_test'], data['y_test']
    
    return (x_train[k], y_train[k]), (x_test[k], y_test[k])

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

if __name__ == "__main__":
    #import numpy as np
    import os
    #from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    #from keras.utils.vis_utils import plot_model

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lam_recon', default=0.225, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=5, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result/PDNA-224-oversample')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # load data
    #(x_train, y_train), (x_test, y_test) = load_PDNA543_hhm()
    traindatafile = 'PDNA224_HHM_7.npz'

    
    y_ps, y_ts = np.zeros((0,2)), np.zeros((0,2))
    for i in range(1):
        (x_train, y_train), (x_test, y_test) = load_data(k=i, npzfile='PDNA224_HHM_7_kfold10_resampling.npz')
        x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)
        x_train, y_train = shuffle(x_train, y_train)
        # ensembling various kernel size
        y_pred = np.zeros(shape=y_test.shape)
        kers = [3,5,7,9,11,13,15]
        for ker in kers:
            # define model
            model = CapsNet(input_shape=x_train.shape[1:],
                            n_class=len(np.unique(np.argmax(y_train, 1))),
                            num_routing=args.num_routing, kernel_size=ker, 
                            padding='same', prim_cap_dim=8, 
                            dig_cap_dim=16, conv_filters=64
                            )
            model.summary()
        
            train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
            
            # load best weight on model
            model.load_weights(args.save_dir + '/trained_model.h5')
            
            y_p = test(model=model, data=(x_test, y_test))
            y_pred += y_p
        
        y_pred = y_pred/len(kers)
        
        K.clear_session()
        tf.reset_default_graph()
 
    
        writeMetrics('PDNA224_oversample_result.txt', y_test, y_pred, 'Fold-{} Predicted Metrics:'.format(i))
 
        y_ps = np.concatenate((y_ps, y_pred))
        y_ts = np.concatenate((y_ts, y_test))
    writeMetrics('PDNA224_oversample_result.txt', y_ts, y_ps, 'Total Predicted Metrics:'.format(i))
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