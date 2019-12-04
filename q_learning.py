import pandas as pd 
import numpy as np 
import time

"""
Qlearning is an off policy learning python implementation.
This is a python implementation of the qlearning algorithm in the Sutton and
Barto's book on RL. It's called SARSA because - (state, action, reward, state,
action). The only difference between SARSA and Qlearning is that SARSA takes the
next action based on the current policy while qlearning takes the action with
maximum utility of next state.
Using the simplest gym environment for brevity: https://gym.openai.com/envs/FrozenLake-v0/
"""

def init_q(s, a, type="ones"):
    """
    @param s the number of states
    @param a the number of actions
    @param type random, ones or zeros for the initialization
    """
    if type == "ones":
        return np.ones((s, a))
    elif type == "random":
        return np.random.random((s, a))
    elif type == "zeros":
        return np.zeros((s, a))


def epsilon_greedy(Q, epsilon, n_actions, s, train=False):
    """
    @param Q Q values state x action -> value
    @param epsilon for exploration
    @param s number of states
    @param train if true then no random actions selected
    """
    if train or np.random.rand() < epsilon:
        action = np.argmax(Q[s, :])
    else:
        action = np.random.randint(0, n_actions)
    return action

def qlearning(trajs, n_states, n_actions, alpha, gamma, epsilon):
    """
    @param alpha learning rate
    @param gamma decay factor
    @param epsilon for exploration
    @param max_steps for max step in each episode
    @param n_tests number of test episodes
    """
    # env = gym.make('Taxi-v2')
    # n_states, n_actions = env.observation_space.n, env.action_space.n

    Q = init_q(n_states, n_actions, type="zeros")
    timestep_reward = []
    # for episode in range(episodes):
    for traj in trajs:
        s = traj[0][0]
        a = epsilon_greedy(Q, epsilon, n_actions, s)
        total_reward = 0
        # done = False
    
        for t in range(len(traj)):
        # while t < max_steps:
            # s_, reward, done, info = env.step(a)
            s_ = traj[t][3]
            reward = traj[t][2]
            total_reward += reward
            a_ = np.argmax(Q[s_, :])
            Q[s, a] += alpha * ( reward + (gamma * Q[s_, a_]) - Q[s, a] )
            s, a = s_, a_

        timestep_reward.append(total_reward)

    return Q, timestep_reward


