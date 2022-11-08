from torch import nn
import torch
import gym
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

#Qua abbiamo 2 cose da tenere in mente: Il replay_buffer ha una dimensione molto ampia, però il training della rete non avviene quando l'intero buffer è pieno, ma quando si raggiunge
#un "minimo di presenze" all'interno del buffer, dal valore espresso in MIN_REPLAY_SIZE. A quel punto si fa il training sulla batch impostata.

env = gym.make("CartPole-v1")

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self,x):
        return self.net(x)

    def act(self,obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize Replay Buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    env.step(action)

    new_obs, rew, done, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs
    if done:
        obs = env.reset()

# Main Training Loop

obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    #bella funzione per interpolare la Epsilon e farla decadere dall'episodio iniziale fino ad un certo episodio al valore da noi voluto con un decay in un numero definito di step.
    #Bisogna passare pure lo step

    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, rew, done, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += rew
    # After solved, watch it play
    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) >= 500:
            while True:
                action = online_net.act(obs)

                obs, _, done, _=env.step(action)
                env.render()
                if done:
                    env.reset()
    if done:
        obs = env.reset()

        rew_buffer.append(episode_reward)
        episode_reward= 0.0

    # Start Gradient Step
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    # Compute Targets
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rews_t + GAMMA * (1-dones_t) * max_target_q_values
    #In questo caso se l'episodio è terminale, l'osservazione successiva è con reward 0 e quindi non possiamo prendere alcuna azione, di conseguenza annulliamo tutto il resto e
    #lasciamo solo il reward appena raccolto senza il q_value per l'azione da intraprendere.
    # Compute Loss
    q_values = online_net(obses_t)

    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
    #Questa parte un po strana, non l'ho capita, anzichè usare Max usa torch.gather, molto strano
    loss = nn.functional.smooth_l1_loss(action_q_values,targets)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # logging
    if step % 1000 == 0:
        print()
        print("Step", step)
        print("Avg Rew", np.mean(rew_buffer))