import os

if __name__ == '__main__':
    print('gamma start')
    for i in range(5):
        print(f'gamma-{0.95 + i * 0.01} start')
        os.system(f'python PPO_main.py --gamma {0.95 + i * 0.01}')

    print('lambda start')
    for i in range(5):
        print(f'lambda-{0.95 + i * 0.01} start')
        os.system(f'python PPO_main.py --lmbda {0.95 + i * 0.01}')

    print('eps start')
    for i in range(8):
        print(f'eps-{0.05 + i * 0.05} start')
        os.system(f'python PPO_main.py --eps {0.05 + i * 0.05}')

    print('episodes start')
    for i in range(10):
        print(f'episodes-{100000 + i * 100000} start')
        os.system(f'python PPO_main.py --num_episodes {100000 + i * 100000}')