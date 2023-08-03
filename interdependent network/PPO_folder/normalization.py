import numpy as np


class RunningMeanStd:
    """ 用于计算 均值 和 标准差 """
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    """ 提供标准化方法 """
    def __init__(self, shape):
        self.running = RunningMeanStd((shape, 1))

    def __call__(self, x, update=True):
        if update:
            self.running.update(x)
        x = (x - self.running.mean) / self.running.std  # (self.running.std + 1e-8)
        return x


class RewardScaling:
    """ 提供 reward scaling trick """
    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = float(gamma)
        self.running = RunningMeanStd(shape)
        self.R = np.zeros(shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running.update(self.R)
        x = x / self.running.std  # (self.running.std + 1e-8)
        return x

    def reset(self):
        self.R = np.zeros(self.shape)