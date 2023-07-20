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
        self.stds = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.tanh(self.fc2(x))
        # return F.softmax(self.fc2(x), dim=1)  # TODO: 1 2

class Normal(torch.nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.stds = torch.nn.Parameter(torch.zeros(action_dim)).exp() * 0.05 * math.pi

    def forward(self, x):
        return torch.distributions.Normal(loc=x, scale=self.stds)


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.normal = Normal(action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action_dist = self.normal(self.policy_net(state))
        # probs = self.policy_net(state)
        # action_dist = torch.distributions.Normal(
        #     loc=probs,
        #     scale=torch.nn.Parameter(torch.zeros(state.shape[0])).exp() * 0.05 * math.pi
        # )
        action = action_dist.sample()
        return action.numpy()

    # def take_action(self, state):  # 根据动作概率分布随机采样
    #     state = torch.tensor(state, dtype=torch.float).to(self.device)
    #     probs = self.policy_net(state)
    #     action_dist = torch.distributions.Categorical(probs)
    #     action = action_dist.sample()
    #     return action.numpy()  # TODO:

    # def update(self, transition_dict):
    #     reward_list = transition_dict['rewards']
    #     state_list = transition_dict['states']
    #     action_list = transition_dict['actions']
    #
    #     G = 0
    #     self.optimizer.zero_grad()
    #     for i in reversed(range(len(reward_list))):  # 从最后一步算起
    #         reward = reward_list[i]
    #         state = torch.tensor([state_list[i]],
    #                              dtype=torch.float).to(self.device)
    #         action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
    #         # log_prob = torch.log(self.policy_net(state).gather(1, action))  #TODO:
    #         log_prob = torch.log(self.policy_net(state).squeeze(0).gather(1, action))
    #         G = self.gamma * G + reward
    #         loss = (-log_prob * G).sum()  # 每一步的损失函数
    #         loss.backward()  # 反向传播计算梯度
    #     self.optimizer.step()  # 梯度下降

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']


        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            loss = -self.normal(self.policy_net(state)).log_prob(action) * reward
            loss.backward()
        self.optimizer.step()  # 梯度下降
