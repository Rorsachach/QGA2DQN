import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from REINFORCE import REINFORCE
import env.InterdependentNetworkEnv as Env
from DQN import *
from networks.factory import Factory


def modify_action(action: np.ndarray):
    return [i * 0.05 * math.pi for i in action]


if __name__ == "__main__":
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env = Env.InterdependentNetworkEnv(Factory.generate_interdependent_network())
    # env.seed(0)
    torch.manual_seed(0)

    agent = REINFORCE(2, hidden_dim, 21, learning_rate, gamma,
                      device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state, info = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    real_action = modify_action(action)
                    next_state, reward, done, _, info = env.step(real_action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)