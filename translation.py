#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:30:15 2019

@author: dawei
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
import pandas as pd

np.random.seed(2019)

trajs = pd.read_csv('trajectory_new.csv')
trajs = trajs.loc[trajs['amount'] < 300000]

all_states = ['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']

mapping_list = ['marriage', 'diff_city', 'kids'] 

trajs1_pd = trajs.loc[trajs['labels'] == 0]
trajs2_pd = trajs.loc[trajs['labels'] == 1]


for item in mapping_list:
    x, y = trajs2_pd[all_states].iloc[:100000].values, trajs1_pd[item].iloc[:100000].values
    reg = RandomForestRegressor(n_estimators=10, criterion="mae", random_state=0, verbose=1, n_jobs=-1)
    reg.fit(x.reshape((x.shape[0], len(all_states))), y.reshape((y.shape[0], 1)))
    f_name = './translation/' + item + '.sav'
    pickle.dump(reg, open(f_name, 'wb'))
    print(item, ' done')   
    print(reg.score(x.reshape((x.shape[0], len(all_states))), y.reshape((y.shape[0], 1))))