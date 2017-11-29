import numpy
import layers


class FeedForward:
    def __init__(self, architecture, default=layers.Tanh):
        """
        Create new `FeedForward` NN instance. By default it will append Sigmoid layers

        :param architecture: list of neuron amounts in each layer
        """
        self.layers = []
        self.learn_rate = 0.1
        self.momentum = 0.1
        self.error = 0

        for i in range(0, len(architecture) - 1):
            if i == 0:
                self.layers.append(layers.Linear(architecture[0], architecture[1]))
            else:
                self.layers.append(default(architecture[i], architecture[i + 1]))
        self.layers.append(layers.Linear(architecture[-1], 0))

    def fit(self, x: list, d: list):
        x = numpy.array(x).reshape(len(x), 1)
        x = numpy.vstack(([1], x))
        d = numpy.array(d).reshape(len(d), 1)

        self._forward(x)
        self._backprop(d)
        self._sgd(x, d)

    def _forward(self, x):
        for i in range(1, len(self.layers)):
            l, l_prev = self.layers[i], self.layers[i - 1]
            if i > 1:
                x = numpy.vstack(([1], l_prev.y))
            l.v = l_prev.w.T.dot(x)
            l.y = l.a(l.v)

    def _backprop(self, d):
        for i, l in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1:
                l.delta = l.der(l.v) * (d - l.y)
                self.error = 0.5 * numpy.sum(d - l.y) ** 2
            else:
                l.delta = l.der(l.v) * (l.w[1:, :].dot(self.layers[i + 1].delta))

    def _sgd(self, x, d):
        for i in range(0, len(self.layers) - 1):
            l = self.layers[i]
            if i > 0:
                x = numpy.vstack(([1], l.y))
            l.w += self.learn_rate * x.dot(self.layers[i + 1].delta.T)

    def get(self, x):
        x_ = numpy.array(x).reshape(len(x), 1)
        x_ = numpy.vstack(([1], x_))
        for i in range(1, len(self.layers)):
            l, l_prev = self.layers[i], self.layers[i - 1]
            if i > 1:
                x_ = numpy.vstack(([1], l_prev.y))
            l.v = l_prev.w.T.dot(x_)
            l.y = l.a(l.v)
        return numpy.round(self.layers[-1].y, decimals=3)

    def __add__(self, other: layers.Layer):
        """
        Append layer to `FeedForward` instance

        :param other:
        :return:
        """

    def __repr__(self):
        s = ""
        for l in self.layers:
            s += f"v:\n{l.v}\n\ny:\n{l.y}\n\ndelta:\n{l.delta}\n\nw:\n{l.w}\n\n------------------------------\n"
        return s