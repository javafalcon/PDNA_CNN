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
            conv_filters=128,padding='valid',
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
    conv1 = layers.Conv2D(filters=conv_filters, kernel_size=(5,7), strides=1, padding=padding, activation='relu', name='conv1')(x)

    conv2 = layers.Conv2D(filters=conv_filters, kernel_size=(5,11), strides=1, padding=padding, activation='relu', name='conv2')(x)

    conv3 = layers.Conv2D(filters=conv_filters, kernel_size=(5,15), strides=1, padding=padding, activation='relu', name='conv3')(x)

    conv4 = layers.Conv2D(filters=conv_filters, kernel_size=(5,20), strides=1, padding=padding, activation='relu', name='conv4')(x)

    conv5 = layers.Conv2D(filters=conv_filters, kernel_size=(5,30), strides=1, padding=padding, activation='relu', name='conv5')(x)
    
    #conv = layers.Conv2D(filters=conv_filters, kernel_size=5, strides=1, padding='same', activation='relu', name='conv')(conv)
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps1 = PrimaryCap(conv1, dim_vector=prim_cap_dim, n_channels=dig_cap_dim, kernel_size=5, strides=2, padding=padding, name='pc1')
    primarycaps2 = PrimaryCap(conv2, dim_vector=prim_cap_dim, n_channels=dig_cap_dim, kernel_size=5, strides=2, padding=padding, name='pc2')
    primarycaps3 = PrimaryCap(conv3, dim_vector=prim_cap_dim, n_channels=dig_cap_dim, kernel_size=5, strides=2, padding=padding, name='pc3')
    primarycaps4 = PrimaryCap(conv4, dim_vector=prim_cap_dim, n_channels=dig_cap_dim, kernel_size=5, strides=2, padding=padding, name='pc4')
    primarycaps5 = PrimaryCap(conv5, dim_vector=prim_cap_dim, n_channels=dig_cap_dim, kernel_size=5, strides=2, padding=padding, name='pc5')
    #primarycaps = layers.Concatenate()([primarycaps1, primarycaps2, primarycaps3, primarycaps4, primarycaps5])

    # Layer 3: Capsule layer. Routing algorithm works here.
    #digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=dig_cap_dim, num_routing=num_routing, name='digitcaps')(primarycaps)
    digitcaps1 = CapsuleLayer(num_capsule=n_class, dim_vector=dig_cap_dim, num_routing=num_routing, name='digitcaps1')(primarycaps1)
    digitcaps2 = CapsuleLayer(num_capsule=n_class, dim_vector=dig_cap_dim, num_routing=num_routing, name='digitcaps2')(primarycaps2)
    digitcaps3 = CapsuleLayer(num_capsule=n_class, dim_vector=dig_cap_dim, num_routing=num_routing, name='digitcaps3')(primarycaps3)
    digitcaps4 = CapsuleLayer(num_capsule=n_class, dim_vector=dig_cap_dim, num_routing=num_routing, name='digitcaps4')(primarycaps4)
    digitcaps5 = CapsuleLayer(num_capsule=n_class, dim_vector=dig_cap_dim, num_routing=num_routing, name='digitcaps5')(primarycaps5)
    
    digitcaps = layers.Concatenate()([digitcaps1, digitcaps2, digitcaps3, digitcaps4, digitcaps5])
    
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
    """
    #cw = class_weight.compute_class_weight('balanced',np.unique(y_train), y_train)
    #cw = dict(enumerate(cw))
    model.fit(x_train, y_train, batch_size=args.batch_size, 
              epochs=args.epochs,
              validation_data=[x_test, y_test], 
              #class_weight=cw,
              callbacks=[log, tb, checkpoint, lr_decay])
    """
    """
    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#
    """
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
    parser.add_argument('--lam_recon', default=0.45, type=float)  # 31*30 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=5, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result/PDNA-224')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # load data
    #(x_train, y_train), (x_test, y_test) = load_PDNA543_hhm()
    traindatafile = 'PDNA224_HHM_15.npz'
    
    from dataset224 import load_kf_data
    
    (kf_x_pos_train, kf_x_neg_train), (kf_x_pos_test, kf_x_neg_test)  = load_kf_data(benckmarkFile=traindatafile,k=10)
    y_ps, y_ts = np.zeros((0,2)), np.zeros((0,2))
    N = 3
    for i in range(1):
        (x_test, y_test) = build_test(kf_x_pos_test[i], kf_x_neg_test[i])
        
        (x_train, y_train) = build_resampleTrain(kf_x_pos_train[i], kf_x_neg_train[i], neg_samples=N)
        
        y_pred = np.zeros(shape=(y_test.shape[0],2))
        # define model
        model = CapsNet(input_shape=x_train.shape[1:],
                        n_class=len(np.unique(np.argmax(y_train, 1))),
                        num_routing=args.num_routing, 
                        padding='same', prim_cap_dim=8, 
                        dig_cap_dim=16, conv_filters=32
                        )
        model.summary()
        #plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)
    
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    
        y_pred = test(model=model, data=(x_test, y_test))
        K.clear_session()
        tf.reset_default_graph()
 
        #(x_train, y_train) = build_resampleTrain(kf_x_pos_train[i], kf_x_neg_train[i], neg_samples=N)

        #y_pred = y_pred/len(kers)
        writeMetrics('PDNA224_result.txt', y_test, y_pred, 'Concatenate server kernels(3,7,9,11,15). Fold-{} Predicted Metrics:'.format(i))
 
        #y_ps = np.concatenate((y_ps, y_pred))
        #y_ts = np.concatenate((y_ts, y_test))
   # writeMetrics('PDNA224_result.txt', y_ts, y_ps, 'Total Predicted Metrics:'.format(i))
   