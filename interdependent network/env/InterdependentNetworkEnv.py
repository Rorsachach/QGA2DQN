import math

import gym
import numpy as np

from typing import Optional, Union

import networkx as nx


class InterdependentNetworkEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self, network: nx.Graph, render_mode: Optional[str] = None):
        self.generation_size = 40 # 种群大小

        self.length = 4
        self.render_mode = render_mode

        self.action_counter = 400
        self.state = None

    def step(self, action: list):
        # err_msg = f"{action!r} ({type(action)}) invalid"
        # assert a, err_msg
        assert self.state is not None, "Call reset before using step method"

        self.action_counter += 1

        old_state = self.state.copy()

        for idx, theta in enumerate(action):
            rotate = np.array([
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)]
            ])

            self.state[idx] = rotate.dot(self.state[idx])

        terminated = bool(
            self.action_counter >= 200
            # or
        )

        reward = 0

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.state = random_init_genome(self.length)# 个体编码

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        pass


def random_init_genome(length: int) -> np.ndarray:
    fix = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
    state = np.empty([length, 2])

    for i in range(length):
        theta = np.random.uniform(0, 1) * 90
        theta = math.radians(theta)
        rotate = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)]
        ])

        state[i] = rotate.dot(fix)

    return state