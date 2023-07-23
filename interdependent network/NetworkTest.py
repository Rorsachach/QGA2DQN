import unittest
import env.InterdependentNetworkEnv as Env
from env.InterdependentNetworkEnv import fitness
from networks.factory import Factory

import matplotlib.pyplot as plt

class MyTestCase(unittest.TestCase):
    def test_fitness(self):
        env = Env.InterdependentNetworkEnv(Factory.generate_interdependent_network(), fa=0.2)
        env.reset()

        fitnesses = []
        for i in range(10):
            fitnesses.append(fitness(env.network, env.l, env.L, env.idx2adj, env.state, env.generation_size))

        plt.plot(list(range(10)), fitnesses)
        plt.show()


if __name__ == '__main__':
    unittest.main()
