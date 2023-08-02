import torch
import torch.nn.functional as F
import numpy as np

import math


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.mid = torch.nn.Linear(hidden_dim, hidden_dim / 2)
        self.fc_mu = torch.nn.Linear(hidden_dim / 2, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim / 2, action_dim)
        self.fc_done = torch.nn.Linear(hidden_dim / 2, 2)

    def forward(self, x):
        """
        :param x: 输入的状态
        :return: 输出正太分布的平均值
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.mid(x))
        mu = 0.1 * torch.pi * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        std = torch.diag_embed(std)
        done = F.softmax(self.fc_done(x))
        return mu, std, done
        # return torch.tanh(self.fc2(x))

class Normal(torch.nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.stds = torch.nn.Parameter(torch.eye(action_dim))

    def forward(self, mu, std):
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=std)
        # return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=self.stds * std)
        # return torch.distributions.MultivariateNormal(loc=x * 0.5 * math.pi, covariance_matrix=self.stds)  # 生成一个策略概率密度函数


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma,
                 device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.normal = Normal(action_dim)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).squeeze(dim=1).to(self.device)
        mu, std, done = self.actor(state)
        action_dist = self.normal(mu, std)
        # action_dist = self.normal(self.actor(state))
        action = action_dist.sample()
        return action.numpy()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).squeeze(dim=2).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).squeeze(dim=2).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        mu, std, done = self.actor(states)
        old_log_probs = self.normal(mu.detach(), std.detach()).log_prob(actions)
        # old_log_probs = self.normal(self.actor(states)).log_prob(actions).detach()

        for _ in range(self.epochs):
            mu, std, done = self.actor(states)
            log_probs = self.normal(mu, std).log_prob(actions)
            # log_probs = self.normal(self.actor(states)).log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
