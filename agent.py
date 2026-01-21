from collections import defaultdict
import gymnasium as gym
import numpy as np


class Agent:
    def __init__(self,env,Lambda, lr, init_epsilon, epsilon_decay, final_epsilon, discount_factor):
        self.env = env
        
        #Q[state][action], action = [stick,hit]
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n)) #gives a function that returns zeros for every action for new state keys. Implementing q(s,a)

        self.e_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = lr
        self.Lambda = Lambda

        #exp params:
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon 

        self.discount_factor = discount_factor

        # Track learning progress
        self.training_error = []
        pass

    
    # epsilon greedy policy
    def get_action(self, obs: tuple[int,int,int]): # obs [players currentsum,dealers card,usable ace] returns index of action chosen.
        
        # Explore: epsilon chance
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # exploit: greedy choice for 1- epsilon
        else:
            return int(np.argmax(self.q_values[obs]))


    def Sarsa_update(self, obs, action, reward, terminated, next_obs, next_action):
        
        # 1. Calculate TD Error
        current_q = self.q_values[obs][action]
        
        # If terminal, next Q is 0
        if terminated:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * self.q_values[next_obs][next_action]
            
        TD_error = td_target - current_q

        # 2. Update eligibility trace for current s, a
        self.e_values[obs][action] += 1

        for state in list(self.e_values.keys()):
            self.q_values[state] += self.lr * TD_error * self.e_values[state]
            self.e_values[state] *= self.discount_factor * self.Lambda

        self.training_error.append(TD_error)

    def decay_epsilon(self):
        
        #decay epsilon after each episode
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

