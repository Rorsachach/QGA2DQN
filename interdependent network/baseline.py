import numpy as np
import networkx as nx
import math
from queue import PriorityQueue

from networks.factory import Factory


def init_population(l: int, L: int) -> np.ndarray:
    return np.full([L, 2, 1], [[math.sqrt(1 - l / L)], [math.sqrt(l / L)]])


def randomSolution(l: int, L: int, Population: np.ndarray):
    solution = np.empty(L)
    for idx, bit in enumerate(Population):
        solution[idx] = 1 if np.random.uniform(0, 1) > pow(bit[0], 2) else 0

    length = np.sum(solution == 1)
    if length > l:
        """ 随机删除 """
        idx = np.where(solution == 1)[0]  # 获取 1 的位置
        np.random.shuffle(idx)  # 随机打乱顺序
        chromosome = np.zeros_like(solution)  # 生成 0 数组
        chromosome[idx[:l]] = solution[idx[:l]]  # 将前 l 项进行赋值
    else:  # ...更少
        """ 随机添加 """
        idx = np.where(solution == 0)[0]  # 获取 0 的位置
        np.random.shuffle(idx)  # 随机打乱顺序
        chromosome = np.ones_like(solution)  # 生成 1 数组
        chromosome[idx[:L - l]] = solution[idx[:L - l]]  # 将最后 l - length 项赋值

    return chromosome


def getMegacomponent(network: nx.Graph):
    G1 = network.subgraph([node[0] for node in network.nodes.items() if node[0].startswith('G1')])  # 加边网络
    G2 = network.subgraph([node[0] for node in network.nodes.items() if node[0].startswith('G2')])  # 相依网络

    while True:
        cc1 = list(nx.connected_components(G1))  # 计算加边网络的连通图
        cc2 = list(nx.connected_components(G2))  # 计算相依网络连通图

        cc1.sort(key=len)  # 排序
        cc2.sort(key=len)  # 排序

        cc1 = cc1[:-1]  # 获取所有非最大连通图
        cc2 = cc2[:-1]

        s = set()

        # 获取节点
        for t in cc1:
            for node in t:
                s.add(node)
                s.add(network.nodes[node]['inter_node'])

        for t in cc2:
            for node in t:
                s.add(node)
                s.add(network.nodes[node]['inter_node'])

        # 没有可删除节点
        if len(s) == 0:
            break

        # 删除所有非最大连通图的节点和相依节点
        network.remove_nodes_from(s)

    return len(network.nodes)


def getFitness(network: nx.Graph, Solution: np.ndarray, idx2adj):
    for idx, bit in enumerate(Solution):
        if bit == 1:
            network.add_edge(f'G1-{idx2adj[idx][0]}', f'G1-{idx2adj[idx][1]}')

    network_size = len(network.nodes)

    attack_node_set = list(range(network_size // 2))
    np.random.shuffle(attack_node_set)
    R = 0
    for node in attack_node_set:
        network.remove_node(network.nodes[f'G1-{node}']['inter_node'])  # 删除攻击节点的相依节点
        network.remove_node(f'G1-{node}')  # 删除攻击节点
        R += getMegacomponent(network.copy()) / network_size

    return R / (network_size // 2)


def getIndex(AdjMatrix_initial: np.ndarray):
    res = []
    rows, cols = AdjMatrix_initial.shape
    for i in range(rows):
        for j in range(i + 1, cols):
            if AdjMatrix_initial[i, j] == 0:
                res.append((i, j))

    return res


def getTheta(qubit, bit, b, flag) -> float:
    s = qubit[0] * qubit[1]
    if bit == 0 and b == 1 and flag:
        if s > 0: return -0.05 * np.pi
        elif s < 0 or qubit[0] == 0: return 0.05 * np.pi
    elif bit == 1 and b == 0:
        if not flag:
            if s > 0: return -0.01 * np.pi
            elif s < 0 or qubit[0] == 0: return 0.01 * np.pi
        else:
            if s > 0: return 0.025 * np.pi
            elif s < 0 or qubit[1] == 0: return -0.025 * np.pi
    elif bit == 1 and b == 0:
        if not flag:
            if s > 0: return 0.025 * np.pi
            elif s < 0 or qubit[1] == 0: return -0.025 * np.pi
        else:
            if s > 0: return 0.025 * np.pi
            elif s < 0 or qubit[1] == 0: return -0.025 * np.pi

    return 0.0

def rotate(Population, Solutions, maxSolution, Fitnesses, maxFitness):
    for idx, solution in enumerate(Solutions):
        fitness = Fitnesses[idx]
        for i, bit in enumerate(solution):
            b = maxSolution[i]
            theta = getTheta(Population[i], bit, b, fitness >= maxFitness)
            r = np.array([
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)]
            ])

            Population[i] = r.dot(Population[i])


if __name__ == "__main__":
    network = Factory.generate_interdependent_network()
    G1 = network.subgraph([node[0] for node in network.nodes.items() if node[0].startswith('G1')])
    G2 = network.subgraph([node[0] for node in network.nodes.items() if node[0].startswith('G2')])

    Elite_Population = []
    Elite_Solution = []

    AdjMatrix_initial = np.array(nx.adjacency_matrix(G1).todense())
    idx2adj = getIndex(AdjMatrix_initial)

    L = len(idx2adj)
    l = round(L * 0.6)

    m = 3
    Q = 100
    iter_max = 200

    lmbda = [0.0001, 0.00015, 0.0002]

    Populations = []

    for _ in range(m):
        Population = init_population(l, L)
        Populations.append(Population)
        fitnesses = []
        Solutions = []
        for _ in range(Q):
            Solution = randomSolution(l, L, Population)
            Solutions.append(Solution)
            fitness = getFitness(network.copy(), Solution, idx2adj)
            fitnesses.append(fitness)

        max_fitnesses_idx = sorted(range(Q), key=lambda k: fitnesses[k], reverse=True)[0]
        Elite_Solution.append((fitnesses[max_fitnesses_idx], Solutions[max_fitnesses_idx]))
        Elite_Population.append((fitnesses[max_fitnesses_idx], Population))

    for episode in range(iter_max):
        for i in range(m):
            Population = Populations[i]
            fitnesses = []
            Solutions = []
            for _ in range(Q):
                Solution = randomSolution(l, L, Population)
                Solutions.append(Solution)
                fitness = getFitness(network.copy(), Solution, idx2adj)
                fitnesses.append(fitness)

            max_fitnesses_idx = sorted(range(Q), key=lambda k: fitnesses[k], reverse=True)[0]
            Elite_Solution.append((fitnesses[max_fitnesses_idx], Solutions[max_fitnesses_idx]))
            Elite_Population.append((fitnesses[max_fitnesses_idx], Population))

            rotate(Population, Solutions, Solutions[max_fitnesses_idx], fitnesses, fitnesses[max_fitnesses_idx])

            p = lmbda[i] * (iter_max - episode)

            for idx in range(L):
                if np.random.uniform(0, 1) < p:
                    Population[idx][0], Population[idx][1] = Population[idx][1], Population[idx][0]

            print(Population)


