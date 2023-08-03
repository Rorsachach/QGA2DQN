import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
import argparse

from PPO import PPO
import InterdependentNetworkEnv as Env
from factory import Factory
from normalization import Normalization, RewardScaling


parser = argparse.ArgumentParser(description="超参数设置")
parser.add_argument('--actor_lr',     help='actor 网络学习率，默认值: 1e-3', default=1e-3, type=float)
parser.add_argument('--critic_lr',    help='critic 网络学习率, 默认值: 1e-2', default=1e-2, type=float)
parser.add_argument('--num_episodes', help='训练次数, 默认值: 5000', default=4000, type=float)
parser.add_argument('--hidden_dim',   help='隐藏层维度, 默认值: 128', default=128, type=float)
parser.add_argument('--gamma',        help='折扣率, 默认值: 0.96', default=0.96, type=float)
parser.add_argument('--lmbda',        help='优势估计超参数, 默认值: 0.95', default=0.95, type=float)
parser.add_argument('--epochs',       help='序列训练次数, 默认值: 10', default=10, type=float)
parser.add_argument('--eps',          help='截断范围, 默认值: 0.2', default=0.2, type=float)


def modify_action(action: np.ndarray):
    return [-0.5 * math.pi + i * 0.05 * math.pi for i in action]


def preprocess_state(state: np.ndarray, state_normal):
    s = state[:, 0]
    s = state_normal(s)
    return s


if __name__ == "__main__":
    args = parser.parse_args()

    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    num_episodes = args.num_episodes
    hidden_dim = args.hidden_dim
    gamma = args.gamma
    lmbda = args.lmbda
    epochs = args.epochs
    eps = args.eps
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    torch.manual_seed(0)
    np.random.seed(0)
    env = Env.InterdependentNetworkEnv(Factory.generate_interdependent_network(5), fa=0.2)

    state_dim = env.L
    action_dim = env.L

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
    state_normal = Normalization(state_dim)  # state normalization trick
    reward_scaling = RewardScaling(1, gamma)  # reward scaling trick

    return_list = []
    info_list = []
    mean_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state, info = env.reset()
                state = preprocess_state(state, state_normal)
                reward_scaling.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, info = env.step(action)
                    next_state = preprocess_state(next_state, state_normal)

                    episode_return += reward

                    reward = reward_scaling(reward)[0]
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state

                # print(state)
                return_list.append(episode_return)
                info_list.append(info['fitness'])
                agent.update(transition_dict, i_episode + i * (num_episodes // 10), num_episodes)
                if (i_episode + 1) % 10 == 0:
                    mean_list.append(np.mean(return_list[-10:]))
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
                # print(state)
    # episodes_list = list(range(len(return_list)))
    sns.set()
    sns.lineplot(data=mean_list)
    # plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on Independent Networks')
    plt.savefig(f'./img/returns-alr-{actor_lr}-clr-{critic_lr}-epi-{num_episodes}-hid-{hidden_dim}-gamma-{gamma}-lambda-{lmbda}-epo-{epochs}-eps-{eps}.jpg')
    # plt.show()

    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('DQN on Independent Networks')
    # plt.show()
    #
    # episodes_list = list(range(len(info_list)))
    # plt.plot(episodes_list, info_list)
    sns.lineplot(data=info_list)
    plt.xlabel('Episodes')
    plt.ylabel('fitness')
    plt.title('PPO on Independent Networks')
    plt.savefig(
        f'./img/returns-alr-{actor_lr}-clr-{critic_lr}-epi-{num_episodes}-hid-{hidden_dim}-gamma-{gamma}-lambda-{lmbda}-epo-{epochs}-eps-{eps}.jpg')
    # plt.show()

    # print(env.base_fit)