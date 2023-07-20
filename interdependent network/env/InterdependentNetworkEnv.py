import math

import gym
from gym import spaces
import numpy as np

from typing import Optional, Union

import networkx as nx


class InterdependentNetworkEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self, network: nx.Graph, fa: float = 1, p: float = 1, render_mode: Optional[str] = None):
        self.network = network  # 相依网络
        self.G1 = self.network.subgraph([node[0] for node in self.network.nodes.items() if node[0].startswith('G1')])
        self.G2 = self.network.subgraph([node[0] for node in self.network.nodes.items() if node[0].startswith('G2')])
        self.generation_size = 10  # 种群大小

        self.idx2adj = genome_extraction(self.G1)  # 加边网络侧的空位对应关系
        self.L = len(self.idx2adj)  # 编码长度
        self.l = round(fa * self.L)  # 目标加边数量
        self.render_mode = render_mode

        self.episode = 0  # 回合数
        self.state = None  # genome 基因编码
        self.state_fit = 0  # 适应度

        self.p = p  # 随机攻击保留的节点数量

        self.observation_space = spaces.Sequence(spaces.Box(np.array([0, 0]), np.array([1, 1])))
        self.action_space = spaces.Sequence(spaces.Discrete(21, start=-10))

    def step(self, action: list):
        """
        执行 action
        :param action:
        :return:
        """

        assert self.state is not None, "Call reset before using step method"

        self.episode += 1  # 回合次数 + 1

        for idx, theta in enumerate(action):  # 执行 action，进行状态转移
            rotate = np.array([
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)]
            ])

            self.state[idx] = rotate.dot(self.state[idx])

        # 计算新状态的适应度
        state_fit = fitness(self.network, self.l, self.L, self.idx2adj, self.state, self.generation_size)

        # 终止条件 episode 达到目标，或者适应度收敛
        terminated = bool(
            self.episode >= 100
            # or not any(action)
            # or reward == 0
            # or abs(state_fit - self.state_fit) < 0.01  # TODO: 不应该以此作为终止条件，需要再考虑一下
        )

        # 奖励值：如果适应度增高则 奖励 1，否则不奖励
        # reward = 10 if state_fit >= self.state_fit else -5
        reward = state_fit - self.state_fit if not terminated else state_fit

        self.state_fit = state_fit  # 跟新 state_fit 记录

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {"fitness": self.state_fit}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        """
        环境重置
        :param seed:
        :param options:
        :return:
        """
        super().reset(seed=seed)

        self.state = random_init_genome(self.l, self.L)  # 初始化编码
        self.state_fit = fitness(self.network, self.l, self.L, self.idx2adj, self.state, self.generation_size)
        self.episode = 0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        pass

    def sample(self) -> list:
        return [i * 0.05 * math.pi for i in self.action_space.sample((self.L, None))]


def random_init_genome(l: int, L: int) -> np.ndarray:
    """
    state 初始化操作
    :param l:
    :param L:
    :return:
    """

    fix = np.array([math.sqrt(1 - l / L), math.sqrt(l / L)])
    genome = np.empty([L, 2])

    for i in range(L):
        theta = np.random.uniform(0, 1) * 90
        theta = math.radians(theta)
        rotate = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)]
        ])

        genome[i] = rotate.dot(fix)

    return genome
    # return np.full([L, 2], [math.sqrt(1 - l / L), math.sqrt(l / L)])


def determine(network: nx.Graph, l: int, L: int, gene: np.ndarray, idx2adj: list) -> nx.Graph:
    """
    根据量子编码生成确定性个体
    :param network:
    :param l:
    :param L:
    :param gene:
    :param idx2adj:
    :return:
    """
    # 进行随机坍缩
    res = np.empty(L)
    for idx, bit in enumerate(gene):
        res[idx] = 1 if np.random.uniform(0, 1) > pow(bit[0], 2) else 0

    length = np.sum(res == 1)  # 统计加边个数
    if length > l:  # 如果加边个数比目标加边个数更多
        """ 随机删除 """
        idx = np.where(res == 1)  # 获取 1 的位置
        np.random.shuffle(idx)  # 随机打乱顺序
        chromosome = np.zeros_like(res)  # 生成 0 数组
        chromosome[idx[:l]] = res[idx[:l]]  # 将前 l 项进行赋值
    else:  # ...更少
        """ 随机添加 """
        idx = np.where(res == 0)  # 获取 0 的位置
        np.random.shuffle(idx)  # 随机打乱顺序
        chromosome = np.ones_like(res)  # 生成 1 数组
        chromosome[idx[l - length:]] = res[idx[l - length:]]  # 将最后 l - length 项赋值

    # 将加边添加到相依网络中
    for idx, bit in enumerate(chromosome):
        if bit == 1:
            network.add_edge(f'G1-{idx2adj[idx][0]}', f'G1-{idx2adj[idx][1]}')

    return network


def robustness(network: nx.Graph) -> int:
    """
    计算网络级联失效收敛后的网络大小
    :param network:
    :return:
    """

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


def evaluation(network: nx.Graph):
    """
    评估加边后网络的适应度
    :param network:
    :return:
    """
    res = 0
    # 生成随机攻击序列
    attack_node_set = list(range(5))
    np.random.shuffle(attack_node_set)

    for node in attack_node_set:
        network.remove_node(network.nodes[f'G1-{node}']['inter_node'])  # 删除攻击节点的相依节点
        network.remove_node(f'G1-{node}')  # 删除攻击节点
        res += robustness(network.copy())  # TODO: 计算鲁棒性

    return res / 5


def fitness(network: nx.Graph, l: int, L: int, idx2adj: list, genome: np.ndarray, generation_size: int) -> float:
    """
    计算 network 的适应度
    :param network:
    :param l:
    :param L:
    :param idx2adj:
    :param genome:
    :param generation_size:
    :return:
    """
    fit = 0.0

    for i in range(generation_size):  # 随机生成该基因下的多个个体取平均
        network_ = determine(network.copy(), l, L, genome, idx2adj)  # 随机生成一个该基因下的个体

        r = evaluation(network_)  # 对该个体进行评价

        fit += (r - fit) / (i + 1)  # 计算适应度平均值

    return fit


def genome_extraction(network: nx.Graph) -> list:
    """
    从拓扑中获取编码对应关系
    :param network:
    :return:
    """
    # 转换成邻接矩阵
    edged_network_adj = np.array(nx.adjacency_matrix(network).todense())

    # 遍历邻接矩阵
    res = []
    rows, cols = edged_network_adj.shape
    for i in range(rows):
        for j in range(i + 1, cols):
            if edged_network_adj[i, j] == 0:  # 如果当前没有边，则说明可以加边，进行记录
                res.append((i, j))

    return res
