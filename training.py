from collections import defaultdict
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from agent import Agent
from tqdm import tqdm  # Progress bar
import pickle


#hyper params
n_episodes = 1000000
learning_rate = 0.01       
start_epsilon = 1.0
final_epsilon = 0
epsilon_decay = start_epsilon / (n_episodes / 2)
discount_factor = 1.0
Lambda = 1.0

env = gym.make("Blackjack-v1", sab=True)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = Agent(env,Lambda= 0.8, lr=learning_rate, init_epsilon = start_epsilon, epsilon_decay = epsilon_decay, final_epsilon= final_epsilon, discount_factor= discount_factor)


for episode in tqdm(range(n_episodes)):

    obs, info = env.reset()

    #reset eligiblity trace for the new episode:
    agent.e_values = defaultdict(lambda: np.zeros(env.action_space.n))
    
    #1. inital action
    action = agent.get_action(obs)
    finished = False

    while not finished:
        #2. sample dynamics whats new state
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        #3. Select next action a' here
        next_action = agent.get_action(next_obs)

        # Update using both current and next action
        agent.Sarsa_update(obs, action, reward, terminated, next_obs, next_action)

        finished = terminated or truncated
        
        # Move to next state AND next action
        obs = next_obs
        action = next_action

    agent.decay_epsilon()



#visualise training:
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length

)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)

axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.savefig("training_performance.png")

# save trained agent:
def save_agent(agent,filename = "BlackJack_agent.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(dict(agent.q_values), f)

save_agent(agent)