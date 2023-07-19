import networkx as nx
import numpy as np


class Factory:
    @staticmethod
    def generate_interdependent_network() -> nx.Graph:
        er = nx.generators.erdos_renyi_graph(10, 0.04)
        ba = nx.generators.barabasi_albert_graph(10, 2)

        network = nx.union(er, ba, rename=('G1-', 'G2-'))
        inter_edges = list(range(10))
        np.random.shuffle(inter_edges)

        edges = []
        for idx, val in enumerate(inter_edges):
            edges.append((f'G1-{idx}', f'G2-{val}'))
            network.nodes[f'G1-{idx}']['inter_node'] = f'G2-{val}'
            network.nodes[f'G2-{val}']['inter_node'] = f'G1-{idx}'

        network.add_edges_from(edges)

        return network