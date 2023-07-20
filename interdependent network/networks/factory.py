import networkx as nx
import numpy as np


class Factory:
    @staticmethod
    def generate_interdependent_network(seed=0) -> nx.Graph:
        er = nx.generators.erdos_renyi_graph(5, 0.4, seed=seed)
        ba = nx.generators.barabasi_albert_graph(5, 2, seed=seed)

        network = nx.union(er, ba, rename=('G1-', 'G2-'))
        inter_edges = list(range(5))
        np.random.shuffle(inter_edges)

        edges = []
        for idx, val in enumerate(inter_edges):
            edges.append((f'G1-{idx}', f'G2-{val}'))
            network.nodes[f'G1-{idx}']['inter_node'] = f'G2-{val}'
            network.nodes[f'G2-{val}']['inter_node'] = f'G1-{idx}'

        network.add_edges_from(edges)

        return network