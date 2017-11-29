import unittest
import nn, layers


class TestFeedForward(unittest.TestCase):
    def test_layers_addition(self):
        v = nn.FeedForward()
        v += layers.Linear(2, 3)
        v += layers.Tanh(3, 2)
        v += layers.Linear(2, 1)
        self.assertEqual(len(v.layers), 3)
        self.assertEqual(len(v.layers[1].v), 3)
