# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 07:26:29 2020

@author: lwzjc
"""
import re
import numpy as np
#from transform import TokenAndPositionEmbedding, TransformerBlock
from sklearn import model_selection
from sklearn.utils import class_weight, shuffle
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers, callbacks

def plot_history(history):
    import matplotlib.pyplot as plt
    #%matplotlib inline
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()
    
    plt.clf()   # clear figure    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()
    
# prepare data
def load_data(n_splits=10):
    data = np.load('PDNA_543_train_7.npz')
    train_pos_seqs = data['pos'] 
    train_neg_seqs = data['neg']
    amino_acids = "XARNDCQEGHILKMFPSTWYV"
    regexp = re.compile('[^ARNDCQEGHILKMFPSTWYVX]')
    
    x_pos_train = []
    for seq in train_pos_seqs:
        seq = regexp.sub('X', seq)
        t = []
        for a in seq:
            t.append(amino_acids.index(a))
        x_pos_train.append(t)
        
    x_neg_train = []
    for seq in train_neg_seqs:
        seq = regexp.sub('X', seq)
        t = []
        for a in seq:
            t.append(amino_acids.index(a))
        x_neg_train.append(t)
        
    data = np.load('PDNA_543_test_7.npz')
    test_pos_seqs = data['pos'] 
    test_neg_seqs = data['neg']
    
    x_pos_test = []
    for seq in test_pos_seqs:
        seq = regexp.sub('X', seq)
        t = []
        for a in seq:
            t.append(amino_acids.index(a))
        x_pos_test.append(t)
        
    x_neg_test = []
    for seq in test_neg_seqs:
        seq = regexp.sub('X', seq)
        t = []
        for a in seq:
            t.append(amino_acids.index(a))
        x_neg_test.append(t)
    
    x_train = np.concatenate([x_pos_train, x_neg_train])    
    x_test = np.concatenate([x_pos_test, x_neg_test])
    
    y_pos_train = np.ones(shape=(len(x_pos_train),))
    y_neg_train = np.zeros(shape=(len(x_neg_train),))
    y_train = np.concatenate(([y_pos_train, y_neg_train]))
    
    y_pos_test = np.ones(shape=(len(x_pos_test),))
    y_neg_test = np.zeros(shape=(len(x_neg_test),))
    y_test = np.concatenate(([y_pos_test, y_neg_test]))
    
    sss = model_selection.StratifiedKFold(n_splits=n_splits, 
                                     shuffle=True, 
                                     random_state=43)
    x_trains, x_vals = [], []
    y_trains, y_vals = [], []
    for train_index, test_index in sss.split(x_train, y_train):
        x_t, x_v = x_train[train_index], x_train[test_index]
        y_t, y_v = y_train[train_index], y_train[test_index]
        x_trains.append(x_t)
        x_vals.append(x_v)
        y_trains.append(y_t)
        y_vals.append(y_v)
    return (x_trains, y_trains), (x_vals, y_vals), (x_test, y_test)

def buildModel(maxlen, vocab_size, embed_dim, num_heads, ff_dim, 
               num_blocks, droprate, fl_size, num_classes):
    from nlp_transformer import Encoder, create_padding_mask
    inputs = layers.Input(shape=(maxlen,))
    
    encode_padding_mask = create_padding_mask(inputs)
    encoder = Encoder(n_layers=num_blocks, d_model=embed_dim, n_heads=num_heads, 
                      ffd=ff_dim, input_vocab_size=vocab_size, 
                      max_seq_len=maxlen, dropout_rate=droprate)
    x = encoder(inputs, False, encode_padding_mask)
    
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(droprate)(x)
    x = layers.Dense(fl_size, activation="relu")(x)
    x = layers.Dropout(droprate)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model
def transformer_predictor(model, X_train, y_train, X_test, y_test, modelfile, params):
    keras.backend.clear_session()


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


"""
def build_transformer_model(params:dict):
    vocab_size = params.get('vocab_size', 21)  # Only consider the 21 amino acids:"ARNDCQEGHILKMFPSTWYVX"
    maxlen = params.get('maxlen', 15)  # Only consider the first 1000 amino acids of each protein sequence
    embed_dim = params.get('embed_dim', 20)  # Embedding size for each token
    num_heads = params.get('num_heads', 4)  # Number of attention heads
    ff_dim = params.get('ff_dim', 32)  # Hidden layer size in feed forward network inside transformer
    drop_rate = params.get('drop_rate', 0.1)
    
    keras.backend.clear_session()
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, num_block=5)
    x = transformer_block(x)
    #x = layers.GlobalAveragePooling1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(drop_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    # Train and Evaluate
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def train_model(model, modelFile,
                x_train, x_val, 
                y_train, y_val, 
                batch_size=100, epochs=30):                                        
    my_class_weight = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train).tolist()
    class_weight_dict = dict(zip([x for x in np.unique(y_train)], my_class_weight))
    
    checkpoint = keras.callbacks.ModelCheckpoint(modelFile, monitor='val_loss',
                                       save_best_only=True, 
                                       save_weights_only=True, 
                                       verbose=1)
    history = model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, 
        class_weight = class_weight_dict,
        callbacks = [checkpoint],
        validation_data=(x_val, y_val)
    )
    
    return history
"""
def predict(model, modelFile, x_test):
    model.load_weights(modelFile)
    y_score = model.predict(x_test)
    return y_score

def displayMetrics(y_test, y_score, threshold=0.5):
    y_pred = (y_score > threshold).astype(float)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print("confusion_matrix:\n", cm)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("accuracy:", acc)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    print("MCC:", mcc)
    auc = metrics.roc_auc_score(y_test, y_score)
    print("AUC:", auc)

    
n_splits = 10
(x_trains, y_trains), (x_vals, y_vals), (x_test, y_test) = load_data(n_splits)
"""
#params = {'vocab_size' : 21, 'maxlen' : 15, 'embed_dim' : 15,
#          'num_heads' : 3,   'ff_dim' : 32, 'drop_rate' : 0.1}
y_true = np.zeros(shape=(0,))
score = np.zeros(shape=(0,))
for i in range(n_splits):
    model = build_transformer_model(params)
    modelFile = ".\\save_models\\trainsformer_model_{}".format(i)
    x, y = x_trains[i], y_trains[i]
    x, y = shuffle(x, y)
    history = train_model(model, modelFile, 
                          x_trains[i], x_vals[i], 
                          y_trains[i], y_vals[i],
                          batch_size=100, epochs=30)  
    print("{}-Fold validation: \n".format(i))
    plot_history(history)
    y_score = predict(model, modelFile, x_vals[i])
    displayMetrics(y_vals[i], y_score)
    y_true = np.concatenate([y_true, y_vals[i]])
    score = np.concatenate([score, y_score[:,0]])

print("10-Folds cross-validation result: \n")
displayMetrics(y_true, score)   
"""
# transformer net params
params = {}
params['vocab_size'] = 21
params['maxlen'] = 15
params['embed_dim'] = 20 # Embedding size for each token
params['num_heads'] = 5  # Number of attention heads
params['ff_dim'] = 128  # Hidden layer size in feed forward network inside transformer
params['num_blocks'] = 8
params['droprate'] = 0.2
params['fl_size'] = 64
params['num_classes'] = 2
params['epochs'] = 10
params['batch_size'] = 100
y_true = np.zeros(shape=(0,))
score = np.zeros(shape=(0,))
for i in range(n_splits):
    model = buildModel(params['maxlen'], params['vocab_size'], params['embed_dim'], 
                params['num_heads'], params['ff_dim'],  params['num_blocks'], 
                params['droprate'], params['fl_size'], params['num_classes'])
    model.summary()

    modelFile = ".\\save_models\\trainsformer_model_{}".format(i)
    x, y = x_trains[i], y_trains[i]
    y = keras.utils.to_categorical(y, params['num_classes'])
    x, y = shuffle(x, y)
    y_val = keras.utils.to_categorical(y_vals[i], params['num_classes'])
    transformer_predictor(model, x, y, x_vals[i], y_val, modelFile, params)
    
    y_score = predict(model, modelFile, x_vals[i])
    displayMetrics(np.argmax(y_val,1), np.argmax(y_score, 1))
    y_true = np.concatenate([y_true, y_vals[i]])
    score = np.concatenate([score, y_score[:,0]])

print("10-Folds cross-validation result: \n")
displayMetrics(y_true, score)   
   
