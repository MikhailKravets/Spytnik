"""
Create here layers of neural network
"""
import random

import numpy
import math


class Layer:
    def __init__(self, length, output):
        self.length, self.output = length, output
        self.v = numpy.zeros(shape=(length, 1))
        self.y = numpy.zeros(shape=(length, 1))
        self.delta = numpy.zeros(shape=(length, 1))
        self.prev_delta = numpy.zeros(shape=(length, 1))

        self.w = 2*numpy.random.random(size=(length + 1, output)) - 1
        self.velocity = numpy.zeros(shape=(length + 1, output))

        # you need to define only activation functions by yourself
        self.a = None
        self.der = None


class Linear(Layer):
    def __init__(self, length, output):
        super().__init__(length, output)
        self.a = lambda x: x
        self.der = lambda x: 1


class Sigmoid(Layer):
    def __init__(self, length, output):
        super().__init__(length, output)
        self.a = lambda x: 1.0/(1.0 + numpy.exp(x))
        self.der = lambda x: self.a(x)*(1.0 - self.a(x))


class Tanh(Layer):
    def __init__(self, length, output):
        super().__init__(length, output)
        self.a = lambda x: numpy.tanh(x)
        self.der = lambda x: 1.0 - self.a(x) ** 2


class Relu(Layer):
    def __init__(self, length, output):
        super().__init__(length, output)
        self.a = lambda x: numpy.maximum(x, 0)
        self.der = lambda x: 1 * (x > 0)


class Dropout(Layer):
    def __init__(self, layer: Layer, percentage=0.15):
        super().__init__(layer.length, layer.output)
        self.percentage = percentage
        self.D = numpy.ones(shape=(layer.length, 1))

        self.a = lambda x: self.D * layer.a(x)
        self.der = lambda x: self.D * layer.der(x)
        self.__dropout()

    def __dropout(self):
        del_indices = random.sample(range(0, len(self.y) - 1), int(self.percentage * len(self.y)))
        for r in del_indices:
            self.D[r] = 0
