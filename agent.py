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


    def Sarsa_update(self, env, obs, action, reward, terminated, next_obs):
        
        #1. we have taken an action observe S', R
        #2. choose A' from new S' using epsilon-greedy

        next_action = self.get_action(next_obs)

        #3. calculate TD_error
        
        current_q = self.q_values[obs][action] #estimate of q value

        TD_error = reward + self.discount_factor*self.q_values[next_obs][next_action] - current_q

        #4. Update eligibility trace for S,a

        self.e_values[obs][action] +=1

        #5. Decrease eligibility trace and apply TD_error backwards

        for state in self.q_values:
            for action in range(env.action_space.n):
                eligibility_value = self.e_values[state][action]
                self.q_values[state][action] =  self.q_values[state][action] + self.lr * TD_error * eligibility_value

                self.e_values[state][action] = self.discount_factor * self.Lambda * eligibility_value #Decay eligiblity
