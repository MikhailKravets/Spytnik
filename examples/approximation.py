import matplotlib.pyplot as plot
import random

import numpy

import layers
from core import FeedForward, separate_data
from core.estimators import cv


def f(x):
    return 0.5 * numpy.sin(numpy.exp(x)) - numpy.cos(numpy.exp(-1 * x))


nn = FeedForward(momentum=0.2, learn_rate=0.05, weight_decay=0.2)  # .create([2, 2, 1], default=layers.Tanh)

nn += layers.Tanh(1, 10)
nn += layers.Tanh(10, 10)
nn += layers.Tanh(10, 10)
nn += layers.Tanh(10, 10)

nn += layers.Linear(10, 1)

data = [([x], [f(x)]) for x in numpy.linspace(-2.2, 2.5, 150)]
ts, vs = separate_data(data, 0.15)


# duplicate x and y for easy plotting
x = numpy.linspace(-2.2, 2.5, 150)
y = f(x)

error = []
v_error = []
for i in range(50_000):
    r = random.randint(0, len(ts) - 1)
    nn.fit(ts[r][0], ts[r][1])
    error.append(nn.error)
    if i % 300 == 0:
        v_error.append(cv(nn, vs))

print(nn)

y_trained = []
for v in x:
    y_trained.append(nn.get([v])[0])

plot.subplot(211)
plot.title("f(x) and its approximation")
plot.plot(x, y, label='f(x)')
plot.plot(x, y_trained, label="NN's approximation")
plot.legend()

plot.subplot(212)
plot.title("Training error")
plot.plot(error, label='Training error')
plot.plot([i * 300 for i in range(len(v_error))], v_error, label='Validation error')
plot.legend()
plot.show()