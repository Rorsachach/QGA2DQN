import numpy as np
import math

import env.InterdependentNetworkEnv as nEnv

if __name__ == "__main__":
    env = nEnv.InterdependentNetworkEnv(None)
    state, _ = env.reset()
    action = []
    for i in range(4):
        theta = np.random.uniform(0, 1) * 90
        action.append(math.radians(theta))

    env.step(action)