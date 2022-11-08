import gym
import torch
from collections import deque
from stable_baselines3 import A2C, PPO
# env = gym.make("LunarLander-v2")
# env.reset()
# print("sample action: ", env.action_space.sample())
# print("observation space: ", env.observation_space.shape)
# print("sample observation: ", env.observation_space.sample())
# env.close()
env = gym.make("LunarLander-v2")
env.reset()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
env.reset()
env.close()
