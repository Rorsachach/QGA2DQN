import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import math

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        :param x: 输入的状态
        :return: 输出正太分布的平均值
        """
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class Normal(torch.nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.stds = torch.nn.Parameter(torch.eye(action_dim)) * 0.05 * math.pi

    def forward(self, x):
        return torch.distributions.MultivariateNormal(loc=x, covariance_matrix=self.stds)  # 生成一个策略概率密度函数


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.normal = Normal(action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action_dist = self.normal(self.policy_net(state))  # 通过 state 生成概率密度函数
        action = action_dist.sample()  # 按照概率密度函数生成 state 对应的 action
        return action.numpy()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor(np.array(state_list[i]), dtype=torch.float).to(self.device)
            action = torch.tensor(np.array(action_list[i])).to(self.device)
            # 按照概率
            log_prob = self.normal(self.policy_net(state)).log_prob(action) # pi_theata(s, a): 在状态 state 下，action 数据点对应的概率
            G = self.gamma * G + reward
            loss = - log_prob * G
            loss.backward(retain_graph=True)
        self.optimizer.step()  # 梯度下降