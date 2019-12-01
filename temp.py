# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:30:50 2019

@author: david
"""

import pandas as pd
import keras as K
import numpy as np
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta
from keras.losses import mean_squared_error

def cnn_model_fn(feature_shape):
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

##################
# Given a new dataframe data
        
data.dropna(how='any')
state_action_pairs = data[['gender', 
                            'amount', 
                            'num_loan', 
                            'duration', 
                            'year_ratio', 
                            'diff_city', 
                            'marriage', 
                            'kids', 
                            'month_in', 
                            'housing', 
                            'edu',
                            'motivation',
                            'state_done',
                            'cumulative_overdue_early_difference',
                            'action']].values
rewards = data['reward'].values

# feature standization
for j in range(state_action_pairs.shape[1]):
    mean = np.mean(state_action_pairs[:,j])
    std = np.std(state_action_pairs[:,j])
    print('mean, std of data column %d: %f %f' % (j, mean, std))
    for i in range(state_action_pairs.shape[0]):
        state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
        
# reward standizatioin
print('mean, std of rewards: %f %f' % (np.mean(rewards), np.std(rewards)))   # 88, 209
# reward standization
rewards = (rewards - np.mean(rewards)) / np.std(rewards)

model = cnn_model_fn()        
state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
model.load_weights('./' + '0-reg-0.0110.hdf5')
# no batch if predicting a single data
pred = model.predict(state_action_pairs, batch_size=64)
# final prediction, projected back to original reward scale
pred_final = pred * 208.725478 + 88.320939
