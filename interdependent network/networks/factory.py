import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Factory:
    @staticmethod
    def generate_interdependent_network(size=10, seed=0) -> nx.Graph:
        er = nx.generators.erdos_renyi_graph(size, 0.2, seed=seed)
        ba = nx.generators.barabasi_albert_graph(size, 1, seed=seed)

        network = nx.union(er, ba, rename=('G1-', 'G2-'))
        inter_edges = list(range(size))
        np.random.shuffle(inter_edges)

        edges = []
        for idx, val in enumerate(inter_edges):
            edges.append((f'G1-{idx}', f'G2-{val}'))
            network.nodes[f'G1-{idx}']['inter_node'] = f'G2-{val}'
            network.nodes[f'G2-{val}']['inter_node'] = f'G1-{idx}'

        network.add_edges_from(edges)

        G1 = network.subgraph([node[0] for node in network.nodes.items() if node[0].startswith('G1')])
        G2 = network.subgraph([node[0] for node in network.nodes.items() if node[0].startswith('G2')])

        pos1 = nx.spring_layout(G1, weight=None, scale=3, center=(1, 3))
        pos2 = nx.spring_layout(G2, center=(1, 0))
        pos = dict()
        pos.update(pos1)
        pos.update(pos2)

        nx.draw_networkx_nodes(network, pos=pos1, node_size=10, nodelist=G1.nodes, node_color='r')
        nx.draw_networkx_nodes(network, pos=pos2, node_size=10, nodelist=G2.nodes, node_color='b')
        nx.draw_networkx_edges(network, pos=pos, edgelist=edges, style='dashed')
        nx.draw_networkx_edges(network, pos=pos, edgelist=list(set(network.edges) - set(edges)))

        plt.show()

        return network