import unittest
import spytnik.layers as layers
import spytnik.core as core


class TestLayers(unittest.TestCase):
    def test_sigmoid(self):
        l = layers.Sigmoid(3, 3)
        self.assertEqual(l.a(0), 0.5)
        self.assertEqual(l.der(0), 0.25)

    def test_tanh(self):
        l = layers.Tanh(3, 3)
        self.assertEqual(l.a(0), 0)
        self.assertEqual(l.der(0), 1)

    def test_linear(self):
        l = layers.Linear(3, 3)
        self.assertEqual(l.a(3), 3)
        self.assertEqual(l.der(23), 1)

    def test_relu(self):
        l = layers.Relu(3, 3)
        self.assertEqual(l.a(3), 3)
        self.assertEqual(l.a(-3), 0)
        self.assertEqual(l.der(23), 1)
        self.assertEqual(l.der(-3), 0)

    def test_dropout_drop(self):
        l = layers.Dropout(layers.Linear(10, 6), percentage=0.5)
        zeros = 0
        for row in l.D:
            if row[0] == 0:
                zeros += 1

        self.assertEqual(zeros, len(l.D) // 2)

    def test_dropout_after_training(self):
        n = core.FeedForward(momentum=0.1, learn_rate=0.1)
        drop = layers.Dropout(layers.Tanh(2, 1), percentage=0.5)
        n += layers.Linear(2, 2)
        n += drop
        n += layers.Linear(1, 0)

        s = [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ]

        n.fit(*s[1])
        n.fit(*s[0])
        n.fit(*s[2])
        n.fit(*s[0])
        n.fit(*s[1])

        zeros = 0
        for row in drop.y:
            if row[0] == 0:
                zeros += 1
        self.assertEqual(zeros, len(drop.w) // 2)
