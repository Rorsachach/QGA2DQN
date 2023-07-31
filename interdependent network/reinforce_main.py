import torch
import numpy as np
import matplotlib.pyplot as plt


from REINFORCE import REINFORCE
import env.InterdependentNetworkEnv as Env
from networks.factory import Factory
from tqdm import tqdm


def preprocess_state(state: np.ndarray):
    return state[:, 0]


if __name__ == "__main__":
    # learning_rate = 1e-3
    learning_rate = 0.5
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env = Env.InterdependentNetworkEnv(Factory.generate_interdependent_network(), fa=0.2)
    torch.manual_seed(0)

    agent = REINFORCE(env.L, hidden_dim, env.L, learning_rate, gamma, device)
    return_list = []
    info_list = []
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
                state = preprocess_state(state)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, info = env.step(action)
                    next_state = preprocess_state(next_state)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                info_list.append(info['fitness'])
                agent.update(transition_dict)
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