import pandas as pd 
import numpy as np 
from n_step_bootstrap import off_policy_n_step_sarsa as nsarsa
from q_learning import qlearning
from policy import Policy

# under \Downloads\CS394 RL Theory and Practice\course project

class RandomPolicy(Policy):
    def __init__(self,nA,p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)

    def action_prob(self,state,action=None):
        return self.p[action]

    def action(self,state):
        return np.random.choice(len(self.p), p=self.p)


trajs = pd.read_csv('trajectory_new.csv')
trajs1_pd = trajs.loc[trajs['labels'] == 0]
trajs2_pd = trajs.loc[trajs['labels'] == 1]

trajs1 = []
trajs2 = []

all_states1 = trajs1_pd[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().values.tolist()
all_states2 = trajs2_pd[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().values.tolist()


loan_ids1 = trajs1_pd['loan_id'].drop_duplicates().values.tolist()

for loan_id in loan_ids1:
    df1 = trajs1_pd.loc[trajs1_pd['loan_id'] == loan_id]
    states = df1[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].values.tolist()
    states.insert(0,states[0])
    states_index = []
    for s in states:
        states_index.append(all_states1.index(s))
    actions = df1['action_num'].values.tolist()
    rewards = df1['reward'].values.tolist()

    traj = list(zip(states_index[:-1],actions, rewards, states_index[1:]))
    trajs1.append(traj)


loan_ids2 = trajs2_pd['loan_id'].drop_duplicates().values.tolist()

for loan_id in loan_ids2:
    df2 = trajs2_pd.loc[trajs2_pd['loan_id'] == loan_id]
    states = df2[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].values.tolist()
    states.insert(0,states[0])
    states_index = []
    for s in states:
        states_index.append(all_states2.index(s))
    actions = df2['action_num'].values.tolist()
    rewards = df2['reward'].values.tolist()

    traj = list(zip(states_index[:-1],actions, rewards, states_index[1:]))
    trajs2.append(traj)

nA1 = trajs1_pd['action'].unique().shape[0]
nS1 = trajs1_pd[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().shape[0]

nA2 = trajs2_pd['action'].unique().shape[0]
nS2 = trajs2_pd[['state_done', 'cumulative_overdue_early_difference','gender','amount','num_loan','duration','year_ratio','diff_city','marriage','kids','month_in','housing','edu','motivation']].drop_duplicates().shape[0]

gamma = 1. # discount factor

behavior_policy1 = RandomPolicy(nA1)
sarsa_Q_star_est1, sarsa_pi_star_est1 = nsarsa(gamma,trajs1,behavior_policy1, nS = nS1, nA = nA1, n=1,alpha=0.005)
print('###########################################################################')
print(sarsa_Q_star_est1)
print('estimated policy is')
print(sarsa_pi_star_est1)

behavior_policy2 = RandomPolicy(nA2)
sarsa_Q_star_est2, sarsa_pi_star_est2 = nsarsa(gamma,trajs2,behavior_policy2, nS = nS2, nA = nA2, n=1,alpha=0.005)
print('###########################################################################')
print(sarsa_Q_star_est2)
print('estimated policy is')
print(sarsa_pi_star_est2)

qlearning1 = qlearning(trajs1, nS1, nA1, alpha = 0.4, gamma = 0.999, epsilon = 0.9)
print('###########################################################################')
print(qlearning1)
qlearning2 = qlearning(trajs2, nS2, nA2, alpha = 0.4, gamma = 0.999, epsilon = 0.9)
print('###########################################################################')
print(qlearning2)

