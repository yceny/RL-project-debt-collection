# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:00:04 2019

@author: david
"""

from numpy.random import seed
from tensorflow import set_random_seed
import numpy as np
import pandas as pd
import keras as K
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling1D
from keras.optimizers import SGD, Adadelta
from keras.losses import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

seed(0)
set_random_seed(0)

#%%
"""
Network architecture

In: feature shape (1-D)
Out: Compiled model
"""

def cnn_model_fn(feature_shape):
        model = Sequential()
        #model.add(Conv1D(32, 4, strides=1, activation='relu', padding="same", input_shape=(feature_shape,1)))
        #model.add(Conv1D(64, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        #model.add(Conv1D(128, 4, strides=1, activation='linear', kernel_initializer= 'normal', padding="same"))
        #model.add(Conv1D(128, 4, strides=1, activation='linear', kernel_initializer= 'normal', padding="same"))


        #model.add(Conv1D(256, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        #model.add(Conv1D(256, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        #model.add(Conv1D(256, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        
        #model.add(Conv1D(512, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        #model.add(Conv1D(512, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        #model.add(Conv1D(512, 4, strides=1, activation='relu', kernel_initializer= 'normal', padding="same"))
        #print(model.output.shape)

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
    
#%%
"""
Save model and architecture
"""
def save_model(clf, name):
    clf.save('./' + name + '.hdf5')   # Save model
    yaml_string = clf.to_yaml()
    with open('./' + \
              name + '.yaml', 'w') as f:   # Save architecture
        f.write(yaml_string)
    f.close()


#%%
"""
Load the original dataframe and replace the label columns
"""

#df = pd.read_csv('./trajectory_new.csv')
#df = df[['gender', 
#         'amount',
#         'num_loan',
#         'duration',
#         'year_ratio',
#         'diff_city',
#         'marriage',
#         'kids',
#         'month_in',
#         'housing',
#         'edu',
#         'motivation',
#         'action',
#         'reward']]
#
#for i in range(df.shape[0]):
#    if df.iloc[i, -2] == 'self':
#        df.iloc[i, -2] = 1
#    elif df.iloc[i, -2] == 'family':
#        df.iloc[i, -2] = 2
#    elif df.iloc[i, -2] == 'acquiantance':
#        df.iloc[i, -2] = 3
#    elif df.iloc[i, -2] == 'sms':
#        df.iloc[i, -2] = 4
#    else:
#        df.iloc[i, -2] = 0   # no action
#    if i%100 == 0:
#        print(i)
#    
#df.to_csv('./df_nn.csv')
#%%
# Load dataframe
df = pd.read_csv('./df_nn.csv')
df.dropna(how='any') 
data = df.iloc[:,1:-1].values
labels = df.iloc[:,-1].values
training = False # Training or test?

# feature standization
for j in range(data.shape[1]):
    mean = np.mean(data[:,j])
    std = np.std(data[:,j])
    for i in range(data.shape[0]):
        data[i,j] = (data[i,j] - mean) / std

if training:
    kfold = KFold(n_splits=2, shuffle=True, random_state=0)   # 2 folds
    cvscores = []
    count = 0
    
    # split data

    training_data, training_labels = data[:int(data.shape[0]*0.8)], labels[:int(data.shape[0]*0.8)]
    training_data = np.reshape(training_data, (training_data.shape[0], training_data.shape[1], 1))   # shape:(data_size, f_size, 1)
    print('mean, std of training fold%d: %f %f' % (count, np.mean(labels), np.std(labels)))
    training_labels = (training_labels - np.mean(labels)) / np.std(labels)   # min-max standization
    
    eval_data, eval_labels = data[int(data.shape[0]*0.8):], labels[int(data.shape[0]*0.8):]
    eval_data = np.reshape(eval_data, (eval_data.shape[0], eval_data.shape[1], 1))
    print('mean, std of eva fold%d: %f %f' % (count, np.mean(labels), np.std(labels)))
    eval_labels = (eval_labels - np.mean(labels)) / np.std(labels)
    
    # check shape
    assert training_data.shape[0] == training_labels.shape[0]
    assert eval_data.shape[0] == eval_labels.shape[0]
    
    # training
    model = cnn_model_fn(training_data.shape[1])
    model.fit(training_data, training_labels,   
        batch_size=64,
        epochs=100,
        verbose=2,
        validation_data = (eval_data, eval_labels),
        shuffle=True,
        callbacks=[EarlyStopping(monitor='val_mean_absolute_error', 
                                 patience=3, 
                                 mode='auto')])
        
    # evaluate the model
    scores = model.evaluate(eval_data, eval_labels, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    # Save hdf5 and yaml
    save_model(model, name = './' + str(count) + '-reg' + '-%.4f' %scores[1])
    count += 1
        
    print('Well trained and saved')
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    
else:
    print('mean, std of test set: %f %f' % (np.mean(labels), np.std(labels)))
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    # get prediction
    model.load_weights('./' + '0-reg-0.2548.hdf5')
    pred = model.predict(data, batch_size=64)
#%%
    pred_mod = pred * np.std(labels) + np.mean(labels)
    for i in range(len(pred_mod)):
        if pred_mod[i] < 100:
            pred_mod[i] = 0

