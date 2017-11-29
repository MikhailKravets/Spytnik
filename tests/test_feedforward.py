import unittest
import nn, layers
import random


class TestFeedForward(unittest.TestCase):
    def test_layers_addition(self):
        v = nn.FeedForward()
        v += layers.Linear(2, 3)
        v += layers.Tanh(3, 2)
        v += layers.Linear(2, 1)
        self.assertEqual(len(v.layers), 3)
        self.assertEqual(len(v.layers[1].v), 3)

    def test_by_xor(self):
        n = nn.FeedForward(momentum=0.1, learn_rate=0.1)  # .create([2, 2, 1], default=layers.Tanh)

        n += layers.Linear(2, 2)
        n += layers.Tanh(2, 1)
        n += layers.Linear(1, 0)

        s = [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ]

        E = []

        for i in range(10_000):
            r = random.randint(0, len(s) - 1)
            n.fit(*s[r])
            E.append(n.error)

        for v in s:
            res = n.get(v[0])
            self.assertEqual(v[1][0], res[0])

        for v in s:
            print(n.get(v[0]), end='\n\n')
