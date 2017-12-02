import unittest

import numpy

import core, layers
import random
import os


class TestCore(unittest.TestCase):
    def test_from_csv(self):
        with open('test.csv', 'w+') as file:
            file.write("0,0,-,0\n")
            file.write("1,0,-,1\n")
            file.write("0,1,-,1\n")
            file.write("1,1,-,0\n")
        data = core.from_csv('test.csv')
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0][1][0], 0)
        os.remove('test.csv')

    def test_separation(self):
        data = [
            ([0, 0], [0]),
            ([1, 0], [1]),
            ([0, 1], [1]),
            ([1, 1], [0]),
        ]
        ts, vs = core.separate_data(data, 0.5)
        self.assertEqual(len(ts), len(vs))

    def test_noise(self):
        data = [
            ([0, 0, 0], [1, 1, 1]),
            ([0, 0, 0], [1, 1, 1]),
            ([0, 0, 0], [1, 1, 1]),
            ([0, 0, 0], [1, 1, 1]),
        ]
        core.noise(data, from_range=(10, 20), axis=0)
        for v in data:
            self.assertTrue(v[0][0] > 10)
        core.noise(data, from_range=(10, 20), axis=1)
        for v in data:
            self.assertTrue(v[1][0] > 10)


class TestFeedForward(unittest.TestCase):
    def test_layers_addition(self):
        v = core.FeedForward()
        v += layers.Linear(2, 3)
        v += layers.Tanh(3, 2)
        v += layers.Linear(2, 1)
        self.assertEqual(len(v.layers), 3)
        self.assertEqual(len(v.layers[1].v), 3)

    def test_by_xor(self):
        error = 0.1
        n = core.FeedForward(momentum=0.1, learn_rate=0.1)  # .create([2, 2, 1], default=layers.Tanh)

        n += layers.Linear(2, 2)
        n += layers.Tanh(2, 1)
        n += layers.Linear(1, 0)

        s = [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ]

        for i in range(10_000):
            r = random.randint(0, len(s) - 1)
            n.fit(*s[r])

        for v in s:
            res = n.get(v[0])
            self.assertTrue(abs(v[1][0] - res[0]) < error)

        for v in s:
            print(n.get(v[0]), end='\n\n')

    def test_xor_by_train(self):
        error = 0.1
        n = core.FeedForward(momentum=0.1, learn_rate=0.1)

        n += layers.Linear(2, 2)
        n += layers.Tanh(2, 1)
        n += layers.Linear(1, 0)

        s = [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ]
        n.train(s, 10_000)

        for v in s:
            res = n.get(v[0])
            self.assertTrue(abs(v[1][0] - res[0]) < error)

        for v in s:
            print(n.get(v[0]), end='\n\n')


class TestEnsemble(unittest.TestCase):
    def test_get(self):
        nn1 = core.FeedForward(momentum=0.1, learn_rate=0.1)
        nn1 += layers.Linear(2, 2)
        nn1 += layers.Tanh(2, 2)
        nn1 += layers.Linear(2, 0)

        nn2 = core.FeedForward(momentum=0.1, learn_rate=0.1)
        nn2 += layers.Linear(2, 2)
        nn2 += layers.Tanh(2, 2)
        nn2 += layers.Linear(2, 0)

        ensemble = core.Ensemble([nn1, nn2])
        ensemble.fit([0, 1], [2, 1])

        stack = numpy.vstack((nn1.get([0, 0]), nn2.get([0, 0])))
        self.assertEqual(ensemble.get([0, 0])[0], (stack.sum(axis=0) / len(stack))[0])


class TestEstimators(unittest.TestCase):
    def test_cv(self):
        import core.estimators
        n = core.FeedForward(momentum=0.1, learn_rate=0.1)
        n += layers.Linear(2, 2)
        n += layers.Tanh(2, 1)
        n += layers.Linear(1, 0)

        s = [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ]
        error = core.estimators.cv(n, s)
        self.assertTrue(type(error) is float)