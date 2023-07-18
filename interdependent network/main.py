import torch
import networkx as nx
import numpy as np
import random

import  env.InterdependentNetworkEnv as inEnv

def generate_interdependent_network() -> nx.Graph:
    er = nx.generators.erdos_renyi_graph(100, 0.04)
    ba = nx.generators.barabasi_albert_graph(100, 2)

    inter_net = nx.union(er, ba, rename=('G1-', 'G2-'))
    inter_edges = list(range(100))
    np.random.shuffle(inter_edges)
    inter_edges = [(f'G1-{idx}', f'G2-{val}') for idx, val in enumerate(inter_edges)]

    inter_net.add_edges_from(inter_edges)

    return inter_net

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

    env = inEnv.InterdependentNetworkEnv(generate_interdependent_network(),)
    random.seed(0)
    np.random.seed(0)
    env.reset(seed=0)

    torch.manual_seed(0)
    # replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n