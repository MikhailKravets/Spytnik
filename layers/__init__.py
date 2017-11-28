"""
Create here layers of neural network
"""
import numpy
import math


class Layer:
    def __init__(self, length, output):
        self.v = numpy.zeros(shape=(length, 1))
        self.y = numpy.zeros(shape=(length, 1))
        self.delta = numpy.zeros(shape=(length, 1))
        self.prev_delta = numpy.zeros(shape=(length, 1))

        self.w = 2*numpy.random.random(size=(length + 1, output)) - 1

        # you need to define it by yourself
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
