"""
Create here layers of neural network
"""
import numpy
import math


class Linear:
    def __init__(self, length, prev_length=None):
        self.v = numpy.zeros(shape=(length, 1))
        self.y = numpy.zeros(shape=(length, 1))
        self.delta = numpy.zeros(shape=(length, 1))
        self.prev_delta = numpy.zeros(shape=(length, 1))

        if prev_length is None:
            prev_length = length

        self.w = numpy.random.uniform(-0.9, 0.9, size=(prev_length + 1, length))
        self.a = lambda x: x
        self.der = lambda x: 1


class Sigmoid:
    def __init__(self, length, prev_length=None):
        self.v = numpy.zeros(shape=(length, 1))
        self.y = numpy.zeros(shape=(length, 1))
        self.delta = numpy.zeros(shape=(length, 1))
        self.prev_delta = numpy.zeros(shape=(length, 1))

        if prev_length is None:
            prev_length = length

        self.w = numpy.random.uniform(-0.9, 0.9, size=(prev_length + 1, length))
        self.a = lambda x: 1.0/(1.0 + numpy.exp(x))
        self.der = lambda x: self.a(x)*(1.0 - self.a(x))


class Tanh:
    def __init__(self, length, prev_length=None):
        self.v = numpy.zeros(shape=(length, 1))
        self.y = numpy.zeros(shape=(length, 1))
        self.delta = numpy.zeros(shape=(length, 1))
        self.prev_delta = numpy.zeros(shape=(length, 1))

        self.w = numpy.random.uniform(-0.9, 0.9, size=(prev_length + 1, length))
        self.a = lambda x: numpy.tanh(x)
        self.der = lambda x: 1.0 - self.a(x) ** 2
