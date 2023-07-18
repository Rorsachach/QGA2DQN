import math

import gym
import numpy as np

from typing import Optional, Union

import networkx as nx


class InterdependentNetworkEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self, network: nx.Graph, render_mode: Optional[str] = None):
        self.network = network  # 相依网络
        # self.network_adj = np.array(nx.adjacency_matrix(self.network).todense())
        self.generation_size = 40  # 种群大小

        self.idx2adj = genome_extraction(self.network)  # 加边网络侧的空位对应关系
        self.length = len(self.idx2adj)  # 编码长度
        self.render_mode = render_mode

        self.episode = 0  # 回合数
        self.state = None  # genome 基因编码
        self.state_fit = 0  # 适应度

        self.l = 5
        self.L = 0  # TODO:

    def step(self, action: list):
        """
        执行 action
        :param action:
        :return:
        """
        # err_msg = f"{action!r} ({type(action)}) invalid"
        # assert a, err_msg
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
            self.episode >= 200
            or abs(state_fit - self.state_fit) < 0.01  # TODO: 不应该以此作为终止条件，需要再考虑一下
        )

        # 奖励值：如果适应度增高则 奖励 1，否则不奖励
        reward = 1 if state_fit > self.state_fit else 0

        self.state_fit = state_fit  # 跟新 state_fit 记录

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

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

        self.state = random_init_genome(self.l, self.length)  # 初始化编码
        self.state_fit = fitness(self.network, self.l, self.L, self.idx2adj, self.state, self.generation_size)

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        pass


def random_init_genome(l: int, L: int) -> np.ndarray:
    """
    state 初始化操作
    :param l:
    :param L:
    :return:
    """
    # fix = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
    # state = np.empty([length, 2])
    #
    # for i in range(length):
    #     theta = np.random.uniform(0, 1) * 90
    #     theta = math.radians(theta)
    #     rotate = np.array([
    #         [math.cos(theta), -math.sin(theta)],
    #         [math.sin(theta), math.cos(theta)]
    #     ])
    #
    #     state[i] = rotate.dot(fix)
    #
    # return state

    return np.full([L, 2], [math.sqrt(1 - l / L), math.sqrt(l / L)])


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
    res = np.empty(len(gene))
    for idx, bit in enumerate(gene):
        res[idx] = 1 if np.random.uniform(0, 1) > pow(bit[0], 2) else 0

    # for bit in gene:
    #     tmp = np.random.uniform(0, 1)
    #     res.append(1 if tmp > np.around(pow(bit[0], 2), 2) else 0)
    #
    #
    #
    # if

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

    return size


def evaluation(network: nx.Graph):
    """
    评估加边后网络的适应度
    :param network:
    :return:
    """
    res = 0
    # 生成随机攻击序列
    attack_node_set = list(range(100))
    np.random.shuffle(attack_node_set)

    for node in attack_node_set:
        network.remove_node(f'G1-{node}')
        res += robustness(network)  # TODO: 计算鲁棒性

    return res / 100


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
    # 获取加边网络子图
    node_list = [node[0] for node in network.nodes.items() if node[0].startswith('G1')]
    edged_network = network.subgraph(node_list)

    # 转换成邻接矩阵
    edged_network_adj = np.array(nx.adjacency_matrix(edged_network).todense())

    # 遍历邻接矩阵
    res = []
    rows, cols = edged_network_adj.shape
    for i in range(rows):
        for j in range(i + 1, cols):
            if edged_network_adj[i, j] == 0:  # 如果当前没有边，则说明可以加边，进行记录
                res.append((i, j))

    return res
