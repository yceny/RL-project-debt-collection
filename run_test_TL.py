import pandas as pd 
import numpy as np 
#from n_step_bootstrap import off_policy_n_step_sarsa as nsarsa
from q_learning import qlearning
from policy import Policy
import keras as K
import numpy as np
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling1D
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD, Adadelta
from keras.losses import mean_squared_error
import wittgenstein as lw 
import pickle
from sklearn.ensemble import RandomForestClassifier

#%%

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

np.random.seed(2019)

trajs = pd.read_csv('trajectory_new.csv')
trajs = trajs.loc[trajs['amount'] < 300000]

idx = np.random.rand(len(trajs)) < 0.6
trajs_train = trajs[idx]
trajs_test = trajs[~idx]

all_states = trajs_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().values.tolist()

trajectory = []

trajs1_pd = trajs.loc[trajs['labels'] == 0]
trajs2_pd = trajs.loc[trajs['labels'] == 1]

idx1 = np.random.rand(len(trajs1_pd)) < 0.5
trajs1_pd_train = trajs1_pd[idx1]
trajs1_pd_test = trajs1_pd[~idx1]

idx2 = np.random.rand(len(trajs2_pd)) < 0.5
trajs2_pd_train = trajs2_pd[idx2]
trajs2_pd_test = trajs2_pd[~idx2]

trajs1 = []
trajs2 = []

all_states1 = trajs1_pd_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().values.tolist()
all_states2 = trajs2_pd_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().values.tolist()


################################################################# all ################################################################################


#loan_ids = trajs_train['loan_id'].drop_duplicates().values.tolist()
#
#
#for loan_id in loan_ids:
#    df = trajs_train.loc[trajs_train['loan_id'] == loan_id]
#    states = df[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].values.tolist()
#    states.insert(0,states[0])
#    states_index = []
#    for s in states:
#        states_index.append(all_states.index(s))
#    actions = df['action_num'].values.tolist()
#    rewards = df['reward'].values.tolist()
#
#    traj = list(zip(states_index[:-1], actions, rewards, states_index[1:]))
#    trajectory.append(traj)
#
#
#
#
#nA = trajs_train['action_num'].unique().shape[0]
#nS = trajs_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().shape[0]
#
## gammas = [1., 0.9, 0.8, 0.7]
## epsilons = [0.9, 0.8]
#
#gamma = 0.7
#epsilon = 0.8
#
#behavior_policy = RandomPolicy(nA)
#sarsa_Q_star_est, sarsa_reward, sarsa_pi_star_est = nsarsa(gamma,trajectory,behavior_policy, nS = nS, nA = nA, n=1,alpha=0.005, epsilon = epsilon)
#
#opt_action_sarsa = []
#for i in range(sarsa_Q_star_est.shape[0]):
#    opt_action_sarsa.append(np.argmax(sarsa_Q_star_est[i,:]))
#
#df_final = trajs_train[['state_done','cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates()
#df_final['opt_action_sarsa'] = opt_action_sarsa
#
#qlearning_star_est, qLearning_reward = qlearning(trajectory, nS, nA, alpha = 0.4, gamma = gamma, epsilon = epsilon)
#opt_action_qlearning = []
#for i in range(qlearning_star_est.shape[0]):
#    opt_action_qlearning.append(np.argmax(qlearning_star_est[i,:]))
#
#df_final['opt_action_qlearning'] = opt_action_qlearning
#
#trajs_test = trajs_test.merge(df_final, on = ['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation'], how = 'left')
## print('shape of test data ', trajs_test.shape)
#
#trajs_test = trajs_test.dropna(how='any')
## print('shape of test data after drop na ', trajs_test.shape)
#
## print('# of installment ', trajs_test[['loan_id','installment']].drop_duplicates().shape)
## print('# of loans ', trajs_test['loan_id'].drop_duplicates().shape)
#
#
## estimate reward for sarsa
## print('######################### estimate reward for sarsa ###############################################')
#state_action_pairs = trajs_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','opt_action_sarsa']].values
#rewards = trajs_test['reward'].values
#
## feature standization
#for j in range(state_action_pairs.shape[1]):
#    mean = np.mean(state_action_pairs[:,j])
#    std = np.std(state_action_pairs[:,j])
#    # print('mean, std of data column %d: %f %f' % (j, mean, std))
#    for i in range(state_action_pairs.shape[0]):
#        state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
#        
## reward standizatioin
#reward_mean = np.mean(rewards)
#reward_std = np.std(rewards)
## print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
## reward standization
#rewards = (rewards - reward_mean) / reward_std
#
#model = cnn_model_fn()        
#state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
#model = load_model('./' + '0-reg-0.0594.hdf5')
## no batch if predicting a single data
#pred = model.predict(state_action_pairs, batch_size=64)
## final prediction, projected back to original reward scale
#pred_final = pred * reward_std + reward_mean
#
#trajs_test['est_reward_sarsa'] = pred_final
#
## estimate reward for q learning
## print('######################### estimate reward for q learning ###############################################')
#state_action_pairs = trajs_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','opt_action_qlearning']].values
#rewards = trajs_test['reward'].values
#
## feature standization
#for j in range(state_action_pairs.shape[1]):
#    mean = np.mean(state_action_pairs[:,j])
#    std = np.std(state_action_pairs[:,j])
#    # print('mean, std of data column %d: %f %f' % (j, mean, std))
#    for i in range(state_action_pairs.shape[0]):
#        state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
#        
## reward standizatioin
#reward_mean = np.mean(rewards)
#reward_std = np.std(rewards)
## print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
## reward standization
#rewards = (rewards - reward_mean) / reward_std
#
#model = cnn_model_fn()        
#state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
#model = load_model('./' + '0-reg-0.0594.hdf5')
## no batch if predicting a single data
#pred = model.predict(state_action_pairs, batch_size=64)
## final prediction, projected back to original reward scale
#pred_final = pred * reward_std + reward_mean
#
#trajs_test['est_reward_qlearning'] = pred_final
#
## actual reward under NN
## print('######################### reward under NN ###############################################')
#state_action_pairs = trajs_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','action_num']].values
#rewards = trajs_test['reward'].values
#
## feature standization
#for j in range(state_action_pairs.shape[1]):
#    mean = np.mean(state_action_pairs[:,j])
#    std = np.std(state_action_pairs[:,j])
#    # print('mean, std of data column %d: %f %f' % (j, mean, std))
#    for i in range(state_action_pairs.shape[0]):
#        state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
#        
## reward standizatioin
#reward_mean = np.mean(rewards)
#reward_std = np.std(rewards)
## print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
## reward standization
#rewards = (rewards - reward_mean) / reward_std
#
#model = cnn_model_fn()        
#state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
#model = load_model('./' + '0-reg-0.0594.hdf5')
## no batch if predicting a single data
#pred = model.predict(state_action_pairs, batch_size=64)
## final prediction, projected back to original reward scale
#pred_final = pred * reward_std + reward_mean
#
#trajs_test['reward_nn'] = pred_final
#
## generate random action and estimate reward for random action
#trajs_test['random_action'] = np.random.randint(0, 5, trajs_test.shape[0])
#
## print('######################### reward under random action ###############################################')
#state_action_pairs = trajs_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','random_action']].values
#rewards = trajs_test['reward'].values
#
## feature standization
#for j in range(state_action_pairs.shape[1]):
#    mean = np.mean(state_action_pairs[:,j])
#    std = np.std(state_action_pairs[:,j])
#    # print('mean, std of data column %d: %f %f' % (j, mean, std))
#    for i in range(state_action_pairs.shape[0]):
#        state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
#        
## reward standizatioin
#reward_mean = np.mean(rewards)
#reward_std = np.std(rewards)
## print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
## reward standization
#rewards = (rewards - reward_mean) / reward_std
#
#model = cnn_model_fn()        
#state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
#model = load_model('./' + '0-reg-0.0594.hdf5')
## no batch if predicting a single data
#pred = model.predict(state_action_pairs, batch_size=64)
## final prediction, projected back to original reward scale
#pred_final = pred * reward_std + reward_mean
#
#trajs_test['est_reward_random'] = pred_final
#
## trajs_test.to_csv('res.csv')
#
#print('############################## result for all ##################################################')
#print('total reward actual ', trajs_test['reward'].sum())
#print('total reward actual nn ', trajs_test['reward_nn'].sum())
#print('total reward for sarsa ', trajs_test['est_reward_sarsa'].sum())
#print('total reward for qlearning ', trajs_test['est_reward_qlearning'].sum())
#print('total reward for random action ', trajs_test['est_reward_random'].sum())




################################################################# group 1 ############################################################################

loan_ids1 = trajs1_pd_train['loan_id'].drop_duplicates().values.tolist()
gamma = 0.7
epsilon = 0.8

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
#sarsa_Q_star_est1, sarsa_reward1, sarsa_pi_star_est1 = nsarsa(gamma,trajs1,behavior_policy1, nS = nS1, nA = nA1, n=1,alpha=0.4)

#opt_action1_sarsa = []
#for i in range(sarsa_Q_star_est1.shape[0]):
#     opt_action1_sarsa.append(np.argmax(sarsa_Q_star_est1[i,:]))

df1_final = trajs1_pd_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates()
#df1_final['opt_action_sarsa'] = opt_action1_sarsa

print('type of trajectory is ', type(trajs1))
recommended_actions = []
qlearning1, qLearning1_reward = qlearning(trajs1, nS1, nA1, recommended_Q=[], alpha = 0.4, gamma = gamma, epsilon = 0.9)
opt_action1_qlearning = []
for i in range(qlearning1.shape[0]):
     opt_action1_qlearning.append(np.argmax(qlearning1[i,:]))

df1_final['opt_action_qlearning'] = opt_action1_qlearning

trajs1_pd_test = trajs1_pd_test.merge(df1_final, on = ['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation'], how = 'left')
print('shape of test data ', trajs1_pd_test.shape)

trajs1_pd_test = trajs1_pd_test.dropna(how='any')
print('shape of test data after drop na ', trajs1_pd_test.shape)

print('# of installment ', trajs1_pd_test[['loan_id','installment']].drop_duplicates().shape)
print('# of loans ', trajs1_pd_test['loan_id'].drop_duplicates().shape)
 # print(trajs1_pd_test.columns.values)

 # estimate reward for sarsa
print('######################### estimate reward for sarsa ###############################################')
#state_action_pairs = trajs1_pd_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','opt_action_sarsa']].values
#rewards = trajs1_pd_test['reward'].values
#
# # feature standization
#for j in range(state_action_pairs.shape[1]):
#     mean = np.mean(state_action_pairs[:,j])
#     std = np.std(state_action_pairs[:,j])
#     print('mean, std of data column %d: %f %f' % (j, mean, std))
#     for i in range(state_action_pairs.shape[0]):
#         state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
#        
# # reward standizatioin
#reward_mean = np.mean(rewards)
#reward_std = np.std(rewards)
#print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
# # reward standization
#rewards = (rewards - reward_mean) / reward_std
#
#model = cnn_model_fn()        
#state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
#model = load_model('./' + '0-reg-0.0594.hdf5')
# # no batch if predicting a single data
#pred = model.predict(state_action_pairs, batch_size=64)
# # final prediction, projected back to original reward scale
#pred_final = pred * reward_std + reward_mean
#
#trajs1_pd_test['est_reward_sarsa'] = pred_final

 # estimate reward for q learning
print('######################### estimate reward for q learning ###############################################')
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
reward_mean = np.mean(rewards)
reward_std = np.std(rewards)
print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
 # reward standization
rewards = (rewards - reward_mean) / reward_std

model = cnn_model_fn()        
state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
model = load_model('./' + '0-reg-0.0594.hdf5')
 # no batch if predicting a single data
pred = model.predict(state_action_pairs, batch_size=64)
 # final prediction, projected back to original reward scale
pred_final = pred * reward_std + reward_mean

trajs1_pd_test['est_reward_qlearning'] = pred_final

 # actual reward under NN
print('######################### reward under NN ###############################################')
state_action_pairs = trajs1_pd_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','action_num']].values
rewards = trajs1_pd_test['reward'].values

 # feature standization
for j in range(state_action_pairs.shape[1]):
     mean = np.mean(state_action_pairs[:,j])
     std = np.std(state_action_pairs[:,j])
     print('mean, std of data column %d: %f %f' % (j, mean, std))
     for i in range(state_action_pairs.shape[0]):
         state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
        
 # reward standizatioin
reward_mean = np.mean(rewards)
reward_std = np.std(rewards)
print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
 # reward standization
rewards = (rewards - reward_mean) / reward_std

model = cnn_model_fn()        
state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
model = load_model('./' + '0-reg-0.0594.hdf5')
 # no batch if predicting a single data
pred = model.predict(state_action_pairs, batch_size=64)
 # final prediction, projected back to original reward scale
pred_final = pred * reward_std + reward_mean

trajs1_pd_test['reward_nn'] = pred_final

# # generate random action and estimate reward for random action
#trajs1_pd_test['random_action'] = np.random.randint(0, 5, trajs1_pd_test.shape[0])

#print('######################### reward under random action ###############################################')
#state_action_pairs = trajs1_pd_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','random_action']].values
#rewards = trajs1_pd_test['reward'].values
#
# # feature standization
#for j in range(state_action_pairs.shape[1]):
#     mean = np.mean(state_action_pairs[:,j])
#     std = np.std(state_action_pairs[:,j])
#     print('mean, std of data column %d: %f %f' % (j, mean, std))
#     for i in range(state_action_pairs.shape[0]):
#         state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
#        
# # reward standizatioin
#reward_mean = np.mean(rewards)
#reward_std = np.std(rewards)
#print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
# # reward standization
#rewards = (rewards - reward_mean) / reward_std
#
#model = cnn_model_fn()        
#state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
#model = load_model('./' + '0-reg-0.0594.hdf5')
# # no batch if predicting a single data
#pred = model.predict(state_action_pairs, batch_size=64)
# # final prediction, projected back to original reward scale
#pred_final = pred * reward_std + reward_mean
#
#trajs1_pd_test['est_reward_random'] = pred_final

trajs1_pd_test.to_csv('./res_group1.csv')

print('############################## result for group 1 ##################################################')
print('total reward actual ', trajs1_pd_test['reward'].sum())
print('total reward actual nn ', trajs1_pd_test['reward_nn'].sum())
#print('total reward for sarsa ', trajs1_pd_test['est_reward_sarsa'].sum())
print('total reward for qlearning ', trajs1_pd_test['est_reward_qlearning'].sum())
#print('total reward for random action ', trajs1_pd_test['est_reward_random'].sum())

#%%
########################################################### Rule learning ##########################################################################
rule_states = trajs1_pd_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference']]
rule_actions = trajs1_pd_test.opt_action_qlearning

      
clf = RandomForestClassifier(n_estimators=100, random_state=0, verbose=1, n_jobs=-1)
print('begin to learn Ds..')
clf.fit(rule_states, rule_actions)
print('rule score: ', clf.score(rule_states, rule_actions))
f_name = './translation/' + 'rule_learner.sav'
pickle.dump(clf, open(f_name, 'wb'))
clf = pickle.load(open(f_name, 'rb'))

#%%
########################################################### group 2 ##########################################################################

loan_ids2 = trajs2_pd_train['loan_id'].drop_duplicates().values.tolist()

for loan_id in loan_ids2:
     df2 = trajs2_pd_train.loc[trajs2_pd_train['loan_id'] == loan_id]
     states = df2[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].values.tolist()
     states.insert(0,states[0])
     states_index = []
     for s in states:
         states_index.append(all_states2.index(s))
     actions = df2['action_num'].values.tolist()
     rewards = df2['reward'].values.tolist()

     traj = list(zip(states_index[:-1],actions, rewards, states_index[1:]))
     trajs2.append(traj)

nA2 = trajs2_pd_train['action_num'].unique().shape[0]
nS2 = trajs2_pd_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().shape[0]


behavior_policy2 = RandomPolicy(nA2)
# sarsa_Q_star_est2, sarsa_reward2, sarsa_pi_star_est2 = nsarsa(gamma,trajs2,behavior_policy2, nS = nS2, nA = nA2, n=1,alpha=0.4)

# opt_action2_sarsa = []
# for i in range(sarsa_Q_star_est2.shape[0]):
#     opt_action2_sarsa.append(np.argmax(sarsa_Q_star_est2[i,:]))

df2_final = trajs2_pd_train[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates()
#df2_final['opt_action_sarsa'] = opt_action2_sarsa

# transfer
print('begin translation')
mapping_list = ['marriage', 'diff_city', 'kids'] 
state_list = ['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']
df2_final_new = df2_final.copy()
for item in mapping_list:
    f_name = './translation/2-1/' + item + '.sav'
    reg = pickle.load(open(f_name, 'rb'))
    x_test = df2_final[state_list]
    translation = reg.predict(x_test)
    df2_final_new[item] = translation
print('get translation')

#%%    
rule_states2 = df2_final_new[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference']]
print('begin to predict recommended actions')
pred = clf.predict(rule_states2)
df2_final_new['recommended_actions'] = pred
    
loan_ids2_t = trajs2_pd_train['loan_id'].drop_duplicates().values.tolist()
trajs2_t = []

for loan_id in loan_ids2_t:
     df2_t = trajs2_pd_train.loc[trajs2_pd_train['loan_id'] == loan_id]
     states = df2_t[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].values.tolist()
     states.insert(0,states[0])
     states_index = []
     for s in states:
         states_index.append(all_states2.index(s))
     actions = df2_final_new['recommended_actions'].values.tolist()

     traj = list(zip(states_index[:-1],actions))
     trajs2_t.append(traj)   
trajs2_t_list = [state_action for traj_item in trajs2_t for state_action in traj_item ]
     
#%%
random_trajs2 = np.random.choice(trajs2, 1, replace=False)

qlearning2, qlearning2_reward = qlearning(trajs2, nS2, nA2, recommended_Q=[], alpha = 0.4, gamma = gamma, epsilon = 0.9)
qlearning2_transfer, qlearning2_reward_transfer = qlearning(trajs2, nS2, nA2, recommended_Q=trajs2_t_list, alpha = 0.4, gamma = gamma, epsilon = 0.9)

    
opt_action2_qlearning = []
for i in range(qlearning2.shape[0]):
     opt_action2_qlearning.append(np.argmax(qlearning2[i,:]))
     
opt_action2_qlearning_tf = []
for i in range(qlearning2_transfer.shape[0]):
     opt_action2_qlearning_tf.append(np.argmax(qlearning2_transfer[i,:]))

df2_final['opt_action_qlearning'] = opt_action2_qlearning
df2_final['opt_action_qlearning_tf'] = opt_action2_qlearning_tf


trajs2_pd_test_copy = trajs2_pd_test.merge(df2_final, on = ['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation'], how = 'left')
print('shape of test data ', trajs2_pd_test_copy.shape)

trajs2_pd_test_copy = trajs2_pd_test_copy.dropna(how='any')
print('shape of test data after drop na ', trajs2_pd_test_copy.shape)

print('# of installment ', trajs2_pd_test_copy[['loan_id','installment']].drop_duplicates().shape)
print('# of loans ', trajs2_pd_test_copy['loan_id'].drop_duplicates().shape)

# # estimate reward for sarsa
# print('######################### estimate reward for sarsa ###############################################')
# state_action_pairs = trajs2_pd_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','opt_action_sarsa']].values
# rewards = trajs2_pd_test['reward'].values
#
# # feature standization
# for j in range(state_action_pairs.shape[1]):
#     mean = np.mean(state_action_pairs[:,j])
#     std = np.std(state_action_pairs[:,j])
#     print('mean, std of data column %d: %f %f' % (j, mean, std))
#     for i in range(state_action_pairs.shape[0]):
#         state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
#        
# # reward standizatioin
# reward_mean = np.mean(rewards)
# reward_std = np.std(rewards)
# print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
# # reward standization
# rewards = (rewards - reward_mean) / reward_std
#
# model = cnn_model_fn()        
# state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
# model = load_model('./' + '0-reg-0.0594.hdf5')
# # no batch if predicting a single data
# pred = model.predict(state_action_pairs, batch_size=64)
# # final prediction, projected back to original reward scale
# pred_final = pred * reward_std + reward_mean
#
# trajs2_pd_test['est_reward_sarsa'] = pred_final

 # estimate reward for q learning
print('######################### estimate reward for q learning ###############################################')
state_action_pairs = trajs2_pd_test_copy[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference', 'opt_action_qlearning']].values
state_action_pairs_tf = trajs2_pd_test_copy[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference', 'opt_action_qlearning_tf']].values
rewards = trajs2_pd_test_copy['reward'].values


 # feature standization
for j in range(state_action_pairs.shape[1]):
     mean = np.mean(state_action_pairs[:,j])
     std = np.std(state_action_pairs[:,j])
     print('mean, std of data column %d: %f %f' % (j, mean, std))
     for i in range(state_action_pairs.shape[0]):
         state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
for j in range(state_action_pairs_tf.shape[1]):
     mean = np.mean(state_action_pairs_tf[:,j])
     std = np.std(state_action_pairs_tf[:,j])
     print('mean, std of data column %d: %f %f' % (j, mean, std))
     for i in range(state_action_pairs_tf.shape[0]):
         state_action_pairs_tf[i,j] = (state_action_pairs_tf[i,j] - mean) / std
        
# reward standizatioin
reward_mean = np.mean(rewards)
reward_std = np.std(rewards)
print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
 # reward standization
rewards = (rewards - reward_mean) / reward_std

model = cnn_model_fn()        
state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
state_action_pairs_tf = np.reshape(state_action_pairs_tf, (state_action_pairs_tf.shape[0], state_action_pairs_tf.shape[1], 1))
model = load_model('./' + '0-reg-0.0594.hdf5')
 # no batch if predicting a single data
pred = model.predict(state_action_pairs, batch_size=64)
pred_tf = model.predict(state_action_pairs_tf, batch_size=64)
 # final prediction, projected back to original reward scale
pred_final = pred * reward_std + reward_mean
pred_tf_final = pred_tf * reward_std + reward_mean

trajs2_pd_test_copy['est_reward_qlearning'] = pred_final
trajs2_pd_test_copy['est_reward_qlearning_tf'] = pred_tf_final

 # actual reward under NN
print('######################### actual reward under NN ###############################################')
state_action_pairs = trajs2_pd_test_copy[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','action_num']].values
rewards = trajs2_pd_test_copy['reward'].values

 # feature standization
for j in range(state_action_pairs.shape[1]):
     mean = np.mean(state_action_pairs[:,j])
     std = np.std(state_action_pairs[:,j])
     print('mean, std of data column %d: %f %f' % (j, mean, std))
     for i in range(state_action_pairs.shape[0]):
         state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
        
 # reward standizatioin
reward_mean = np.mean(rewards)
reward_std = np.std(rewards)
print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
 # reward standization
rewards = (rewards - reward_mean) / reward_std

model = cnn_model_fn()        
state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
model = load_model('./' + '0-reg-0.0594.hdf5')
 # no batch if predicting a single data
pred = model.predict(state_action_pairs, batch_size=64)
 # final prediction, projected back to original reward scale
pred_final = pred * reward_std + reward_mean

trajs2_pd_test_copy['reward_nn'] = pred_final

# # generate random action and estimate reward for random action
# trajs2_pd_test['random_action'] = np.random.randint(0, 5, trajs2_pd_test.shape[0])
#
# print('######################### reward under random action ###############################################')
# state_action_pairs = trajs2_pd_test[['gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation','state_done', 'cumulative_overdue_early_difference','random_action']].values
# rewards = trajs2_pd_test['reward'].values
#
# # feature standization
# for j in range(state_action_pairs.shape[1]):
#     mean = np.mean(state_action_pairs[:,j])
#     std = np.std(state_action_pairs[:,j])
#     print('mean, std of data column %d: %f %f' % (j, mean, std))
#     for i in range(state_action_pairs.shape[0]):
#         state_action_pairs[i,j] = (state_action_pairs[i,j] - mean) / std
#        
# # reward standizatioin
# reward_mean = np.mean(rewards)
# reward_std = np.std(rewards)
# print('mean, std of rewards: %f %f' % (reward_mean, reward_std))   # 88, 209
# # reward standization
# rewards = (rewards - reward_mean) / reward_std
#
# model = cnn_model_fn()        
# state_action_pairs = np.reshape(state_action_pairs, (state_action_pairs.shape[0], state_action_pairs.shape[1], 1))
# model = load_model('./' + '0-reg-0.0594.hdf5')
# # no batch if predicting a single data
# pred = model.predict(state_action_pairs, batch_size=64)
# # final prediction, projected back to original reward scale
# pred_final = pred * reward_std + reward_mean
#
# trajs2_pd_test['est_reward_random'] = pred_final

trajs2_pd_test_copy.to_csv('res_group2.csv')

print('############################## result for group 2 ##################################################')
print('total reward actual ', trajs2_pd_test_copy['reward'].sum())
print('total reward actual nn ', trajs2_pd_test_copy['reward_nn'].sum())
# print('total reward for sarsa ', trajs2_pd_test['est_reward_sarsa'].sum())
print('total reward for qlearning ', trajs2_pd_test_copy['est_reward_qlearning'].sum())
print('total reward for qlearning + tf ', trajs2_pd_test_copy['est_reward_qlearning_tf'].sum())
# print('total reward for random action ', trajs2_pd_test['est_reward_random'].sum())