import unittest
import layers


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