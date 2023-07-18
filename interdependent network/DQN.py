import numpy as np
import math

import env.InterdependentNetworkEnv as nEnv

import networkx as nx


def generate_interdependent_network() -> nx.Graph:
    er = nx.generators.erdos_renyi_graph(100, 0.04)
    ba = nx.generators.barabasi_albert_graph(100, 2)

    inter_net = nx.union(er, ba, rename=('G1-', 'G2-'))
    inter_edges = list(range(100))
    np.random.shuffle(inter_edges)
    inter_edges = [(f'G1-{idx}', f'G2-{val}') for idx, val in enumerate(inter_edges)]

    inter_net.add_edges_from(inter_edges)

    return inter_net


if __name__ == '__main__':

    inter_net = generate_interdependent_network()

    env = nEnv.InterdependentNetworkEnv(inter_net)
    state, _ = env.reset()
    action = []
    for i in range(4):
        theta = np.random.uniform(0, 1) * 90
        action.append(math.radians(theta))

    env.step(action)
