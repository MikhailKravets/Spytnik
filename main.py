"""
TODO: use this file only to run the code.
Move the NN classes to another package
"""
import numpy
import layers


class FeedForward:
    def __init__(self, architecture):
        """
        Create new `FeedForward` NN instance. By default it will append Sigmoid layers

        :param architecture: list of neuron amounts in each layer
        """
        self.layers = []
        self.learn_rate = 0.1
        self.momentum = 0.1
        self.error = 0

        for i in range(1, len(architecture)):
            self.layers.append(layers.Sigmoid(architecture[i], architecture[i - 1]))

    def fit(self, x: list, d: list):
        x.insert(0, 1)
        x = numpy.array(x).reshape(len(x), 1)
        d = numpy.array(d).reshape(len(d), 1)

        # straight way
        for i, l in enumerate(self.layers):
            if i > 0:
                x = numpy.vstack((numpy.array([1]), self.layers[i - 1].y))
            l.v = l.w.T.dot(x)
            l.y = l.a(l.v)

        # backprop
        for i, l in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1:
                l.delta = l.der(l.v) * (d - l.y)
            else:
                l.delta = l.der(l.v) * (self.layers[i + 1].w[1:, :].dot(self.layers[i + 1].delta))

        # SGD
        for l in self.layers:
            pass

    def train(self,):
        pass

    def get(self, x):
        pass

    def __repr__(self):
        s = ""
        for l in self.layers:
            s += f"v:\n{l.v}\n\ny:\n{l.y}\n\ndelta:\n{l.delta}\n\nw:\n{l.w}\n\n------------------------------\n"
        return s

nn = FeedForward([2, 2, 1])
print(nn)
nn.fit([0, 1], [1])
print(nn)
