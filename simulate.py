import gymnasium as gym
import pickle
from agent import Agent
from collections import defaultdict
import numpy as np
import time

env = gym.make("Blackjack-v1", render_mode="human")

# Load the trained Q-table
with open("BlackJack_agent.pkl", "rb") as f:
    trained_q_values = pickle.load(f)


agent = Agent(env,0,0,0,0,0,0)
agent.q_values = defaultdict(lambda: np.zeros(env.action_space.n), trained_q_values)


for episode in range(10):
    obs, info = env.reset()
    done = False
    print(f"\nEpisode {episode}")

    while not done:
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(1.0)


    #print reward after episode
    print("Reward:", reward)

env.close()