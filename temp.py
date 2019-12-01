#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:39:56 2019

@author: dawei
"""

import pandas as pd
import keras as K
import numpy as np
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta
from keras.losses import mean_squared_error

def cnn_model_fn():
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(64, activation ='relu')) # Do not add batchnorm in dense, break linear
        #model.add(Dropout(0.2))
        model.add(Dense(64, activation ='relu'))   # may Overfit if too complicated
        #model.add(Dropout(0.2))
        model.add(Dense(128, activation ='relu', kernel_initializer= 'normal'))
        model.add(Dense(128, activation ='relu', kernel_initializer= 'normal'))
        #model.add(Dense(256, activation ='relu', kernel_initializer= 'normal'))
        #model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer= 'normal'))
            
        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        loss = mean_squared_error
        model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['mae'])
        return model
    
data.dropna(how='any')
state_action_pairs = data.iloc[:,0:-1].values
rewards = data.iloc[:,-1].values
# feature standization
for j in range(state_action_pairs.shape[1]):
    mean = 88
    std = 209
    for i in range(state_action_pairs.shape[0]):
        state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std

model = cnn_model_fn()        
state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
model.load_weights('./' + '0-reg-0.2548.hdf5')
pred = model.predict(state_action_pairs)
# project back to original reward scale
pred_mod = pred * np.std(rewards) + np.mean(rewards)
for i in range(len(pred_mod)):
    if pred_mod[i] < 100:
        pred_mod[i] = 0
    
