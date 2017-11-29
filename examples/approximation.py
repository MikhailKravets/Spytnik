import matplotlib.pyplot as plot
import random

import numpy

import layers
from nn import FeedForward


def f(x):
    return 0.5 * numpy.sin(numpy.exp(x)) - numpy.cos(numpy.exp(-1 * x))


nn = FeedForward(momentum=0.1, learn_rate=0.05)  # .create([2, 2, 1], default=layers.Tanh)

nn += layers.Linear(1, 7)

nn += layers.Tanh(7, 8)
nn += layers.Tanh(8, 7)
nn += layers.Tanh(7, 1)

nn += layers.Linear(1, 0)

x = numpy.linspace(-1.6, 2.5, 150)
y = f(x)

error = []

for i in range(50_000):
    r = random.randint(0, len(x) - 1)
    nn.fit([x[r]], [y[r]])
    error.append(nn.error)

print(nn)

y_trained = []
for v in x:
    y_trained.append(nn.get([v])[0])

plot.subplot(211)
plot.title("f(x) and its approximation")
plot.plot(x, y)
plot.plot(x, y_trained)

plot.subplot(212)
plot.title("Learning error")
plot.plot(error)
plot.show()