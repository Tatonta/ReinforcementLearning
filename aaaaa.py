import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision
from torch import nn
from collections import deque
import itertools
import numpy as np
import random

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
TARGET_UPDATE_FREQ=1000

env = gym.make("CartPole-v1")

observations = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    all = env.step(action)

    if done:
        observation, info = env.reset()
env.close()