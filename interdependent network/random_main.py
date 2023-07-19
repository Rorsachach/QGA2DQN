import torch
import networkx as nx
import numpy as np
import random

import env.InterdependentNetworkEnv as Env

import matplotlib.pyplot as plt
from networks.factory import Factory


if __name__ == "__main__":
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env = Env.InterdependentNetworkEnv(Factory.generate_interdependent_network(), )
    random.seed(0)
    np.random.seed(0)
    env.reset(seed=0)

    action = env.sample()

    return_list = []

    for i in range(10):
        env.reset()
        done = False
        episode_return = 0
        times = 0
        while not done:
            action = env.sample()
            next_state, reward, done, _, info = env.step(action)
            episode_return += info['fitness']
            print(f'times: {times}')
            times += 1
        print(f'Iterator {i} is done.')
        return_list.append(episode_return / times)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.show()
