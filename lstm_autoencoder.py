# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:54:57 2021

@author: shais
"""

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import   Masking
from keras.preprocessing.sequence import pad_sequences

class LSTM_Autoencoder(object):
    """docstring for LSTM_Autoencoder"""
    def __init__(self):
        self.nb_epoch = 20  # epochs
        self.batch_size = 256
        self.shuffle = True
        self.validation_split = 0.05
        self.optimizer = 'adadelta'
        self.loss = 'mse' 

   
    def baseline_model(self,inputDim, hidden_dim):
        assert len(inputDim) == 2
        inputData = Input(shape=(inputDim[0],inputDim[1])) # make a 'tensor' object for Keras model
        mask = Masking(mask_value=0.)(inputData) # masking model
        encoded = LSTM(hidden_dim)(mask) # LSTM masked layer with mask input
        rep_vector = RepeatVector(inputDim[0])(encoded) ## LSTM masked layer with encoder as input
        decoded = LSTM(inputDim[1], return_sequences=True)(rep_vector) # return_sequences=True for return hidden state vector
        self.model = Model(inputData, decoded)
        

    def compile(self, *args):
        if len(args) == 0:
            self.model.compile(optimizer=self.optimizer, loss=self.loss)
        elif len(args) == 1 and args[0] == 'temporal':
            #timestep-wise sample weighting (2D weights)
            self.sample_weight_mode = args[0]
            self.model.compile(optimizer=self.optimizer, loss=self.loss, sample_weight_mode=self.sample_weight_mode)
        else: 
            raise ValueError("argument # must be 0 or ='temporal' (sampleWeights mask type)")


    def fit(self, *args):
        X_train = args[0]
        y_train = args[0] # on autoencoder we expect the output to be the same as input
        if len(args) == 2:	#without sampleWeights
            if args[1] == 'normal':
                self.model.fit(X_train,
                               y_train,
                               epochs=self.nb_epoch, 
                               batch_size=self.batch_size, 
                               shuffle=self.shuffle, 
                               validation_split=self.validation_split)
            elif args[1] == 'reverse':
                self.model.fit(X_train, 
                               np.flip(y_train, 1), 
                               epochs=self.nb_epoch, 
                               batch_size=self.batch_size, 
                               shuffle=self.shuffle, 
                               validation_split=self.validation_split)
            else: 
                raise ValueError("decoding sequence type: 'normal' or 'reverse'.")

        elif len(args) == 3: #with sampleWeights
            self.sampleWeights = args[2]
            if args[1] == 'normal':
                self.model.fit(X_train,
                               y_train,
                               epochs=self.nb_epoch, 
                               batch_size=self.batch_size, 
                               shuffle=self.shuffle, 
                               validation_split=self.validation_split, 
                               sample_weight=self.sampleWeights)
            elif args[1] == 'reverse':
                self.model.fit(X_train, 
                               np.flip(y_train, 1), 
                               epochs=self.nb_epoch, 
                               batch_size=self.batch_size, 
                               shuffle=self.shuffle, 
                               validation_split=self.validation_split,
                               sample_weight=self.sampleWeights)
            else: 
                raise ValueError("Please input, 'data', 'nor' or 'rev', 'sample_weights'")

    def predict(self, data):
        return self.model.predict(data)


def gen_hid_repre(fea_dim, hid_dim, step_length):
    
    """
    :param fea_dim: input dimension of LSTM-AE model
    :param hid_dim: output dimension of hidden representation
    :return: fixed-length hidden representation of editing sequence.
    """
    
    x_ben = np.load('data/wiki/X_v8_4_50_Ben.npy', encoding='bytes',allow_pickle=True)
    x_van = np.load('data/wiki/X_v8_4_50_Van.npy', encoding='bytes',allow_pickle=True)
    
    train_ben = x_ben[0:7000]

    sampleWeights = list()

    for e in train_ben:
        sampleWeights.append(np.ones(len(e)))
        
    # decoding sequence is reversed
    sampleWeights = pad_sequences(sampleWeights, maxlen=50, dtype='float', padding='post')

    train_ben_P = pad_sequences(train_ben, maxlen=50, dtype='float')
    x_ben_P = pad_sequences(x_ben, maxlen=50, dtype='float')
    x_van_P = pad_sequences(x_van, maxlen=50, dtype='float')

    timesteps = 50
    input_dim = fea_dim
    
    lstm_autoencoder = LSTM_Autoencoder()
    lstm_autoencoder.baseline_model([timesteps, input_dim], hid_dim)
    lstm_autoencoder.compile('temporal')
    lstm_autoencoder.fit(train_ben_P, 'reverse',sampleWeights)

    hidModel = Sequential()
    hidModel.add(lstm_autoencoder.model.layers[0])
    hidModel.add(lstm_autoencoder.model.layers[1])
    hidModel.add(lstm_autoencoder.model.layers[2])

    ben_hid_emd = hidModel.predict(x_ben_P)
    van_hid_emd = hidModel.predict(x_van_P)
    
    np.save("data/wiki/ben_hid_emd_4_50_8_200_r0.npy",ben_hid_emd)
    np.save("data/wiki/val_hid_emd_4_50_8_200_r0.npy",van_hid_emd)

    return ben_hid_emd, van_hid_emd

ben_hid_emd, van_hid_emd =gen_hid_repre(fea_dim=8, hid_dim=200, step_length=50)
