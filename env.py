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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

seed(0)
set_random_seed(0)

#%%
"""
GBM referance
"""
def RF_model():
    reg = RandomForestRegressor(n_estimators=10, criterion="mae", random_state=0, verbose=1, n_jobs=-1)
    return reg

#%%
"""
Network architecture

In: feature shape (1-D)
Out: Compiled model
"""

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
Load the original dataframe and encode the action column as int
"""
convert = False

if convert:
    df = pd.read_csv('./trajectory_new.csv')
    df = df[['gender', 
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
             'action',
             'reward']]
    
    for i in range(df.shape[0]):
        if df.iloc[i, -2] == 'self':
            df.iloc[i, -2] = 1
        elif df.iloc[i, -2] == 'family':
            df.iloc[i, -2] = 2
        elif df.iloc[i, -2] == 'acquiantance':
            df.iloc[i, -2] = 3
        elif df.iloc[i, -2] == 'sms':
            df.iloc[i, -2] = 4
        else:
            df.iloc[i, -2] = 0   # no action
        if i%100 == 0:
            print(i)
        
    df.to_csv('./df_nn.csv')
#%%
# Load dataframe
df = pd.read_csv('./df_nn.csv')
df.dropna(how='any')
df = df.loc[df['reward'] < 1000]
data = df[['gender', 
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
rewards = df['reward'].values
training = False   # Training or test?
nn = True   # use nn?

# feature standization
for j in range(data.shape[1]):
    mean = np.mean(data[:,j])
    std = np.std(data[:,j])
    print('mean, std of data column %d: %f %f' % (j, mean, std))
    for i in range(data.shape[0]):
        data[i,j] = (data[i,j] - mean) / std
        
# reward standizatioin
print('mean, std of rewards: %f %f' % (np.mean(rewards), np.std(rewards)))
# reward standization
rewards = (rewards - np.mean(rewards)) / np.std(rewards)     

if training:
    kfold = KFold(n_splits=2, shuffle=True, random_state=0)   # 2 folds, may not be used
    cvscores = []
    count = 0
    
    # split data

    training_data, training_labels = data[:int(data.shape[0]*0.8)], rewards[:int(data.shape[0]*0.8)]
    training_data = np.reshape(training_data, (training_data.shape[0], training_data.shape[1], 1))   # shape:(data_size, f_size, 1)  
    
    eval_data, eval_labels = data[int(data.shape[0]*0.8):], rewards[int(data.shape[0]*0.8):]
    eval_data = np.reshape(eval_data, (eval_data.shape[0], eval_data.shape[1], 1))   
    
    # check shape
    assert training_data.shape[0] == training_labels.shape[0]
    assert eval_data.shape[0] == eval_labels.shape[0]
    
    # training
    if nn:
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
    else:
        model = RF_model()
        model.fit(data, rewards)
        R_2 = model.score(data, rewards)
        print("Predicted R^2: %.2f" % (R_2))
        cvscores.append(R_2)
        f_name = './rf' + str(count) + str(R_2) + '.sav'
        pickle.dump(model, open(f_name, 'wb'))
        
    count += 1
        
    print('Well trained and saved')
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    
else:
    # get prediction
    if nn:
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))
        model.load_weights('./' + '0-reg-0.0110.hdf5')
        pred = model.predict(data, batch_size=64)
    else:
        model = pickle.load(open(f_name, 'rb'))
        pred = model.predict(data)
    # project back to original reward scale
    pred_mod = pred * 208.725478 + 88.320939
    # save prediction
    data_prediction = df
    pred_mod = pd.DataFrame(pred_mod) 
    data_prediction['reward'] = pred_mod
    data_prediction.to_csv('./prediction.csv')

            
