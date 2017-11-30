import random
import numpy
import csv

import layers


def from_csv(file_name: str) -> list:
    """
    Load the data from specially formatted .csv file into list of tuples which contain vectors (input data
    and desired output).

    :param file_name: path to `.csv` file
    :return:
    """
    data = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                x, d = list(map(lambda e: float(e), row[:row.index('-')])), \
                       list(map(lambda e: float(e), row[row.index('-') + 1:]))
                data.append((x, d))
            except ValueError:
                pass
    return data


def separate_data(data, percentage: float) -> (list, list):
    """
    Separate the data into training set and validation set with the given percentage

    :param data: list of vectors (input data, desired output)
    :param percentage: float value between [0, 1) which defines how much data move to validation set
    :return: tuple of training set, validation set
    """
    v_len = int(percentage * len(data))
    training, validation = data.copy(), []
    while len(validation) < v_len:
        r = random.randint(0, len(training) - 1)
        validation.append(training[r])
        del training[r]
    return training, validation


def noise(data, from_range=(0, 1), axis=0):
    """
    Add random noise to each element in data from range

    :param data:
    :param from_range:
    :param axis:
    """
    for i in range(len(data)):
        for j in range(len(data[i][axis])):
            data[i][axis][j] += (from_range[1] - from_range[0]) * random.random() + from_range[0]


class FeedForward:
    def __init__(self, learn_rate=0.1, momentum=0.1, weight_decay=0.1):
        """
        Create new `FeedForward` NN instance. By default it will append Sigmoid layers

        :param architecture: list of neuron amounts in each layer
        """
        self.layers = []
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.error = 0

    def fit(self, x, d):
        x = numpy.array(x).reshape(len(x), 1)
        x = numpy.vstack(([1], x))
        d = numpy.array(d).reshape(len(d), 1)

        self._forward(x)
        self._backprop(d)
        self._sgd(x, d)

    def train(self, data, iterations):
        for i in range(iterations):
            r = random.randint(0, len(data) - 1)
            x, d = data[r][0], data[r][1]
            self.fit(x, d)

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
            l.velocity = self.momentum * l.velocity + self.learn_rate * x.dot(self.layers[i + 1].delta.T)
            l.velocity += -self.weight_decay * l.velocity
            l.w += l.velocity

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

    def from_list(self, architecture, default=layers.Tanh):
        self.layers = []
        for i in range(0, len(architecture)):
            self.layers.append(default(architecture[i], architecture[i + 1]))
        return self

    def __add__(self, l: layers.Layer):
        """
        Append layer to `FeedForward` instance

        :param l: the layer to be added to neural network
        :return: link on the current neural network
        """
        if type(l) is layers.Layer:
            raise TypeError("The given layer is not actually 'layer.Layer' instance or its child")

        self.layers.append(l)
        return self

    def __repr__(self):
        s = ""
        for l in self.layers:
            s += f"v:\n{l.v}\n\ny:\n{l.y}\n\ndelta:\n{l.delta}\n\nw:\n{l.w}\n\n------------------------------\n"
        return s