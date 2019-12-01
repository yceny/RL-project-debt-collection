import pandas as pd 
import numpy as np 
from n_step_bootstrap import off_policy_n_step_sarsa as nsarsa
from q_learning import qlearning
from policy import Policy
import keras as K
import numpy as np
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta
from keras.losses import mean_squared_error

class RandomPolicy(Policy):
    def __init__(self,nA,p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)

    def action_prob(self,state,action=None):
        return self.p[action]

    def action(self,state):
        return np.random.choice(len(self.p), p=self.p)

# under \Downloads\CS394 RL Theory and Practice\course project


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


gamma = 1. # discount factor

trajs = pd.read_csv('trajectory_new.csv')
trajs1_pd = trajs.loc[trajs['labels'] == 0]
trajs2_pd = trajs.loc[trajs['labels'] == 1]

idx1 = np.random.rand(len(trajs1_pd)) < 0.8
trajs1_pd_train = trajs1_pd[idx1]
trajs1_pd_test = trajs1_pd[~idx1]

idx2 = np.random.rand(len(trajs2_pd)) < 0.8
trajs2_pd_train = trajs2_pd[idx2]
trajs2_pd_test = trajs2_pd[~idx2]

trajs1 = []
trajs2 = []

all_states1 = trajs1_pd_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().values.tolist()
all_states2 = trajs2_pd_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().values.tolist()


loan_ids1 = trajs1_pd_train['loan_id'].drop_duplicates().values.tolist()

for loan_id in loan_ids1:
    df1 = trajs1_pd_train.loc[trajs1_pd_train['loan_id'] == loan_id]
    states = df1[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].values.tolist()
    states.insert(0,states[0])
    states_index = []
    for s in states:
        states_index.append(all_states1.index(s))
    actions = df1['action_num'].values.tolist()
    rewards = df1['reward'].values.tolist()

    traj = list(zip(states_index[:-1],actions, rewards, states_index[1:]))
    trajs1.append(traj)


nA1 = trajs1_pd_train['action_num'].unique().shape[0]
nS1 = trajs1_pd_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().shape[0]


behavior_policy1 = RandomPolicy(nA1)
sarsa_Q_star_est1, sarsa_reward1, sarsa_pi_star_est1 = nsarsa(gamma,trajs1,behavior_policy1, nS = nS1, nA = nA1, n=1,alpha=0.005)

opt_action1_sarsa = []
for i in range(sarsa_Q_star_est1.shape[0]):
    opt_action1_sarsa.append(np.argmax(sarsa_Q_star_est1[i,:]))

df1_final = trajs1_pd_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates()
df1_final['opt_action_sarsa'] = opt_action1_sarsa

qlearning1, qLearning1_reward = qlearning(trajs1, nS1, nA1, alpha = 0.4, gamma = 0.999, epsilon = 0.9)
opt_action1_qlearning = []
for i in range(qlearning1.shape[0]):
    opt_action1_qlearning.append(np.argmax(qlearning1[i,:]))

df1_final['opt_action_qlearning'] = opt_action1_qlearning

trajs1_pd_test = trajs1_pd_test.merge(df1_final, on = ['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation'], how = 'left')
print('shape of test data ', trajs1_pd_test.shape)

trajs1_pd_test.dropna(how='any')
print('shape of test data after drop na ', trajs1_pd_test.shape)
print(trajs1_pd_test.columns.values)

# estimate reward for sarsa
state_action_pairs = trajs1_pd_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','opt_action_sarsa']].values
rewards = trajs1_pd_test['reward'].values

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

trajs1_pd_test['est_reward_sarsa'] = pred_final

# estimate reward for q learning
state_action_pairs = trajs1_pd_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','opt_action_qlearning']].values
rewards = trajs1_pd_test['reward'].values

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

trajs1_pd_test['est_reward_qlearning'] = pred_final

print('total reward for sarsa ', trajs1_pd_test['est_reward_sarsa'].sum())
print('total reward for qlearning ', trajs1_pd_test['est_reward_qlearning'].sum())




########################################################### group 2 ##########################################################################

# loan_ids2 = trajs2_pd_train['loan_id'].drop_duplicates().values.tolist()

# for loan_id in loan_ids2:
#     df2 = trajs2_pd_train.loc[trajs2_pd_train['loan_id'] == loan_id]
#     states = df2[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].values.tolist()
#     states.insert(0,states[0])
#     states_index = []
#     for s in states:
#         states_index.append(all_states2.index(s))
#     actions = df2['action_num'].values.tolist()
#     rewards = df2['reward'].values.tolist()

#     traj = list(zip(states_index[:-1],actions, rewards, states_index[1:]))
#     trajs2.append(traj)

# nA2 = trajs2_pd_train['action_num'].unique().shape[0]
# nS2 = trajs2_pd_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().shape[0]


# behavior_policy2 = RandomPolicy(nA2)
# sarsa_Q_star_est2, sarsa_reward2, sarsa_pi_star_est2 = nsarsa(gamma,trajs2,behavior_policy2, nS = nS2, nA = nA2, n=1,alpha=0.005)

# opt_action2_sarsa = []
# for i in range(sarsa_Q_star_est2.shape[0]):
#     opt_action2_sarsa.append(np.argmax(sarsa_Q_star_est2[i,:]))

# df2_final = trajs2_pd_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates()
# df2_final['opt_action_sarsa'] = opt_action2_sarsa

# qlearning2, qlearning2_reward = qlearning(trajs2, nS2, nA2, alpha = 0.4, gamma = 0.999, epsilon = 0.9)
# opt_action2_qlearning = []
# for i in range(qlearning2.shape[0]):
#     opt_action2_qlearning.append(np.argmax(qlearning2[i,:]))

# df2_final['opt_action_qlearning'] = opt_action2_qlearning


# trajs2_pd_test = trajs2_pd_test.merge(df2_final, on = ['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation'], how = 'left')
# print('shape of test data ', trajs2_pd_test.shape)

# trajs2_pd_test.dropna(how='any')
# print('shape of test data after drop na ', trajs2_pd_test.shape)

# # estimate reward for sarsa
# state_action_pairs = trajs2_pd_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','opt_action_sarsa']].values
# rewards = trajs2_pd_test['reward'].values
# # feature standization
# for j in range(state_action_pairs.shape[1]):
#     mean = 88
#     std = 209
#     for i in range(state_action_pairs.shape[0]):
#         state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std

# model = cnn_model_fn()        
# state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
# model.load_weights('./' + '0-reg-0.2548.hdf5')
# pred = model.predict(state_action_pairs)
# # project back to original reward scale
# pred_mod = pred * np.std(rewards) + np.mean(rewards)
# for i in range(len(pred_mod)):
#     if pred_mod[i] < 100:
#         pred_mod[i] = 0

# trajs2_pd_test['est_reward_sarsa'] = pred_mod

# # estimate reward for q learning
# state_action_pairs = trajs2_pd_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','opt_action_qlearning']].values
# rewards = trajs2_pd_test['reward'].values
# # feature standization
# for j in range(state_action_pairs.shape[1]):
#     mean = 88
#     std = 209
#     for i in range(state_action_pairs.shape[0]):
#         state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std

# model = cnn_model_fn()        
# state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
# model.load_weights('0-reg-0.0110.hdf5')
# pred = model.predict(state_action_pairs)
# # project back to original reward scale
# pred_mod = pred * np.std(rewards) + np.mean(rewards)
# for i in range(len(pred_mod)):
#     if pred_mod[i] < 100:
#         pred_mod[i] = 0

# trajs2_pd_test['est_reward_qlearning'] = pred_mod

# print('total reward for sarsa ', trajs2_pd_test['est_reward_sarsa'].sum())
# print('total reward for qlearning ', trajs2_pd_test['est_reward_qlearning'].sum())
