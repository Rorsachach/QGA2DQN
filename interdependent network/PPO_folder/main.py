import multiprocessing
import os
from multiprocessing import Process, Pool

class gammaProcess(Process):
    def __init__(self, name, num):
        super(gammaProcess, self).__init__()
        self.name = name
        self.num = num

    def run(self) -> None:
        os.system(f'nohup python PPO_main.py --gamma {0.95 + self.num * 0.01} > /tmp/ppo/gamma-{0.95 + self.num * 0.01}-log 2>&1 &')

class lambdaProcess(Process):
    def __init__(self, name, num):
        super(lambdaProcess, self).__init__()
        self.name = name
        self.num = num

    def run(self) -> None:
        os.system(f'nohup python PPO_main.py --lmbda {0.95 + self.num * 0.01} > /tmp/ppo/lambda-{0.95 + self.num * 0.01}-log 2>&1 &')

class epsProcess(Process):
    def __init__(self, name, num):
        super(epsProcess, self).__init__()
        self.name = name
        self.num = num

    def run(self) -> None:
        os.system(f'nohup python PPO_main.py --eps {0.05 + self.num * 0.05} > /tmp/ppo/eps-{0.05 + self.num * 0.05}-log 2>&1 &')

class episodesProcess(Process):
    def __init__(self, name, num):
        super(episodesProcess, self).__init__()
        self.name = name
        self.num = num

    def run(self) -> None:
        os.system(f'nohup python PPO_main.py --num_episodes {100000 + i * 100000} > /tmp/ppo/episodes-{100000 + i * 100000}-log 2>&1 &')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    pool = Pool(28)
    print('gamma start')
    for i in range(5):
        print(f'gamma-{0.95 + i * 0.01} start')
        # os.system(f'python PPO_main.py --gamma {0.95 + i * 0.01}')
        pool.apply_async(gammaProcess(f'gamma-{0.95 + i * 0.01}', i).start())

    print('lambda start')
    for i in range(5):
        print(f'lambda-{0.95 + i * 0.01} start')
        # os.system(f'python PPO_main.py --lmbda {0.95 + i * 0.01}')
        pool.apply_async(lambdaProcess(f'lambda-{0.95 + i * 0.01}', i).start())

    print('eps start')
    for i in range(8):
        print(f'eps-{0.05 + i * 0.05} start')
        # os.system(f'python PPO_main.py --eps {0.05 + i * 0.05}')
        pool.apply_async(epsProcess(f'eps-{0.05 + i * 0.05}', i).start())

    print('episodes start')
    for i in range(10):
        print(f'episodes-{100000 + i * 100000} start')
        # os.system(f'python PPO_main.py --num_episodes {100000 + i * 100000}')
        pool.apply_async(episodesProcess(f'episodes-{100000 + i * 100000}', i).start())

    pool.close()
    pool.join()
