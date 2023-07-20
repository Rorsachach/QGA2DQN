import env.InterdependentNetworkEnv as Env
from DQN import *
from networks.factory import Factory
import matplotlib.pyplot as plt


flag = False


def modify_action(action: np.ndarray):
    return [-0.05 * math.pi + i * 0.05 * math.pi for i in action]


if __name__ == "__main__":
    lr = 2e-3  # 学习率
    num_episodes = 1000  # 训练次数
    hidden_dim = 128  # 隐藏层维度
    gamma = 0.98  # 折扣因子 [0, 1] 接近 1 关注长期累计奖励，接近 0 关注短期奖励
    epsilon = 0.1  # 引入随机概率，防止陷入贪婪策略的局部最优解
    target_update = 10  # 目标网络更新频率
    buffer_size = 10000  # 回放池大小
    minimal_size = 500  # 最小更新回放池大小
    batch_size = 64  # 随机采样大小
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    # 设置随机发生器种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = Env.InterdependentNetworkEnv(Factory.generate_interdependent_network(), fa=0.2)  # 初始化环境
    replay_buffer = ReplayBuffer(buffer_size)  # 初始化经验回放池
    agent = DQN(2, hidden_dim, 3, lr, gamma, epsilon,
                target_update, device)  # 初始化 agent，这里的 2 是 状态维度，21 是 动作维度

    return_list = []
    info_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:  # 统计 10 轮
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0  # 存放这一轮决策下来的 reward 结果
                state, info = env.reset(seed=0)  # 环境重置
                done = False
                while not done:
                    action = agent.take_action(state)  # agent 进行决策
                    real_action = modify_action(action)  # 从 离散集合 0-21 转向 -0.05pi-0.05pi
                    next_state, reward, done, _, info = env.step(real_action)  # 环境更新
                    replay_buffer.add(state, action, reward, next_state, done)  # 放入经验回放池中
                    state = next_state  # 状态更新
                    episode_return += reward  # 累计奖励更新
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        if not flag:
                            print("begin update.")
                            flag = True
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)  # agent 更新
                return_list.append(episode_return)
                info_list.append(info['fitness'])
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on Independent Networks')
    plt.show()

    episodes_list = list(range(len(info_list)))
    plt.plot(episodes_list, info_list)
    plt.xlabel('Episodes')
    plt.ylabel('fitness')
    plt.title('DQN on Independent Networks')
    plt.show()

    # mv_return = rl_utils.moving_average(return_list, 9)
    # plt.plot(episodes_list, mv_return)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('DQN on {}'.format(env_name))
    # plt.show()
