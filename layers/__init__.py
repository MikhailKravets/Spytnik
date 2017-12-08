"""
Create here layers of neural network
"""
import random

import numpy
import math


class Layer:
    def __init__(self, input_len, output_len):
        self.input, self.output = input_len, output_len
        self.v = numpy.zeros(shape=(output_len, 1))
        self.y = numpy.zeros(shape=(output_len, 1))
        self.delta = numpy.zeros(shape=(output_len, 1))
        self.prev_delta = numpy.zeros(shape=(output_len, 1))

        self.w = 2*numpy.random.random(size=(input_len + 1, output_len)) - 1
        self.velocity = numpy.zeros(shape=(input_len + 1, output_len))

        # you need to define only activation functions by yourself
        self.a = None
        self.der = None


class Input(Layer):
    def __init__(self, input_len, output_len):
        super().__init__(input_len, output_len)
        self.w = numpy.ones(shape=(input_len + 1, output_len))
        self.a = lambda x: x
        self.der = lambda x: 1


class Linear(Layer):
    def __init__(self, input_len, output_len):
        super().__init__(input_len, output_len)
        self.a = lambda x: x
        self.der = lambda x: 1


class Sigmoid(Layer):
    def __init__(self, input_len, output_len):
        super().__init__(input_len, output_len)
        self.a = lambda x: 1.0/(1.0 + numpy.exp(x))
        self.der = lambda x: self.a(x)*(1.0 - self.a(x))


class Tanh(Layer):
    def __init__(self, input_len, output_len):
        super().__init__(input_len, output_len)
        self.a = lambda x: numpy.tanh(x)
        self.der = lambda x: 1.0 - self.a(x) ** 2


class Relu(Layer):
    def __init__(self, input_len, output_len):
        super().__init__(input_len, output_len)
        self.a = lambda x: numpy.maximum(x, 0)
        self.der = lambda x: 1 * (x > 0)


class Dropout(Layer):
    def __init__(self, layer: Layer, percentage=0.15):
        super().__init__(layer.input, layer.output)
        self.percentage = percentage
        self.D = numpy.ones(shape=(layer.output, 1))

        self.a = lambda x: self.D * layer.a(x)
        self.der = lambda x: self.D * layer.der(x)
        self.__dropout()

    def __dropout(self):
        del_indices = random.sample(range(0, len(self.y) - 1), int(self.percentage * len(self.y)))
        for r in del_indices:
            self.D[r] = 0
