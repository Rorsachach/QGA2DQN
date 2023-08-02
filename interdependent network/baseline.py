import random
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    elif bit == 1 and b == 1:
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

    return Population


# def migration(tmp, Elite_Population, Populations):
#     tmp.sort(key=lambda a: a[0], reverse=True)
#     Elite_Population.append(tmp[0])
#
#     if len(Elite_Population) > m:
#         populations = random.sample(Elite_Population, m)
#         for i in range(m):
#             if tmp_Population[i][0] < populations[i][0]:
#                 Populations[i] = populations[i][1]


if __name__ == "__main__":
    """ 生成相依网络 """
    network = Factory.generate_interdependent_network(100)
    # 取出加边侧网络
    G1 = network.subgraph([node[0] for node in network.nodes.items() if node[0].startswith('G1')])
    # 取出相依侧网络
    G2 = network.subgraph([node[0] for node in network.nodes.items() if node[0].startswith('G2')])

    """ 初始化部分 """
    Elite_Population = []  # 用于存储精英染色体库
    Elite_Solution = []  # 用于存储精英解库

    AdjMatrix_initial = np.array(nx.adjacency_matrix(G1).todense())  # 初始邻接矩阵
    idx2adj = getIndex(AdjMatrix_initial)  #

    L = len(idx2adj)  # 可加边网络位置
    l = round(L * 0.2)  # 实际加边个数

    m = 3  # 种群数量
    Q = 100  # 种群空间大小
    iter_max = 200  # 最大迭代次数

    lmbda = [0.0001, 0.00015, 0.0002]  # 编译概率
    np.random.seed(0)

    Populations = []  # 用于存放染色体
    episodes_list = []

    for _ in range(m):
        Population = init_population(l, L)  # 生成初代染色体空间
        Populations.append(Population)  # 保存初代染色体空间
        # 转化解空间，并计算解空间适应度
        fitnesses = []
        solutions = []
        for _ in range(Q):
            solution = randomSolution(l, L, Population)
            solutions.append(solution)
            fitness = getFitness(network.copy(), solution, idx2adj)
            fitnesses.append(fitness)

        elite_solution = sorted(range(Q), key=lambda k: fitnesses[k], reverse=True)[0]  # 获取当前种群的最有个体
        Elite_Solution.append(
            {
                'fitness': fitnesses[elite_solution],
                'solution': solutions[elite_solution],
            })  # 将最优个体保存到精英解库

    elite_population = sorted(range(m), key=lambda k: Elite_Solution[k]['fitness'], reverse=True)[0]
    Elite_Population.append(
        {
            'fitness': Elite_Solution[elite_population]['fitness'],
            'solution': Elite_Solution[elite_population]['solution'],
            'population': Populations[elite_population]
        })  # 添加精英染色体



    with tqdm(total=int(iter_max)) as pbar:
        for episode in range(iter_max):
            for i in range(m):
                Population = Populations[i]
                fitnesses = []
                solutions = []
                for _ in range(Q):
                    solution = randomSolution(l, L, Population)
                    solutions.append(solution)
                    fitness = getFitness(network.copy(), solution, idx2adj)
                    fitnesses.append(fitness)

                rotate(Population, solutions, Elite_Solution[i]['solution'], fitnesses, Elite_Solution[i]['fitness'])

                elite_solution = sorted(range(Q), key=lambda k: fitnesses[k], reverse=True)[0]
                if fitnesses[elite_solution] > Elite_Solution[i]['fitness']:
                    Elite_Solution[i]['fitness'] = fitnesses[elite_solution]
                    Elite_Solution[i]['solution'] = solutions[elite_solution]

                """ 变异 """
                p = lmbda[i] * (iter_max - episode)
                for idx in range(L):
                    if np.random.uniform(0, 1) < p:
                        tmp = Population[idx][0].copy()
                        Population[idx][0] = Population[idx][1]
                        Population[idx][1] = tmp

            # tmp_Population.sort(key=lambda a: a[0], reverse=True)
            # Elite_Population.append(tmp_Population[0])
            #
            # if len(Elite_Population) > m:
            #     populations = random.sample(Elite_Population, m)
            #     for i in range(m):
            #         if tmp_Population[i][0] < populations[i][0]:
            #             Populations[i] = populations[i][1]

            # Elite_Solution.sort(key=lambda a: a[0], reverse=True)
            # print(Elite_Solution[0])
            episodes_list.append(max(Elite_Solution, key=lambda a: a['fitness'])['fitness'])

            if (episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (episode + 1),
                    'return': '%.3f' % episodes_list[episode]
                })
            pbar.update(1)

    print(Populations)

    plt.plot(list(range(iter_max)), episodes_list)
    plt.xlabel('Episodes')
    plt.ylabel('fitness')
    plt.show()

