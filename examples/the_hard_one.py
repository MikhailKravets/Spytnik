"""
This is the hard one. I need to handle it by my own
"""
import random

import matplotlib.pyplot as plot
import time

import numpy

import layers
from core import FeedForward, Ensemble, from_csv, separate_data, noise
from core.estimators import cv


def normal(arr):
    s = numpy.sum(numpy.abs(arr))
    return numpy.abs(arr) / s


training, validation = separate_data(from_csv("D:\\DELETE\\Дипломмо\\output.csv"), 0.15)

# noise(training, from_range=(0, 2), axis=0)
# noise(training, from_range=(-0.05, 0.05), axis=1)

ff1 = FeedForward(learn_rate=0.05, momentum=0.2, weight_decay=0.5)
ff1 += layers.Linear(6, 23)
ff1 += layers.Dropout(layers.Tanh(23, 28), percentage=0.3)
ff1 += layers.Dropout(layers.Tanh(28, 28), percentage=0.3)
ff1 += layers.Dropout(layers.Tanh(28, 28), percentage=0.3)
ff1 += layers.Dropout(layers.Tanh(28, 8), percentage=0.3)
ff1 += layers.Linear(8, 0)

# ff2 = FeedForward(learn_rate=0.07, momentum=0.2, weight_decay=0.23)
# ff2 += layers.Linear(6, 23)
# ff2 += layers.Dropout(layers.Sigmoid(23, 28), percentage=0.3)
# ff2 += layers.Dropout(layers.Sigmoid(28, 28), percentage=0.3)
# ff2 += layers.Dropout(layers.Sigmoid(28, 28), percentage=0.3)
# ff2 += layers.Dropout(layers.Sigmoid(28, 8), percentage=0.3)
# ff2 += layers.Linear(8, 0)
#
# ff3 = FeedForward(learn_rate=0.04, momentum=0.6, weight_decay=0.4)
# ff3 += layers.Linear(6, 23)
# ff3 += layers.Dropout(layers.Sigmoid(23, 28), percentage=0.3)
# ff3 += layers.Dropout(layers.Sigmoid(28, 28), percentage=0.3)
# ff3 += layers.Dropout(layers.Sigmoid(28, 28), percentage=0.3)
# ff3 += layers.Dropout(layers.Sigmoid(28, 8), percentage=0.3)
# ff3 += layers.Linear(8, 0)

ensemble = Ensemble(ff1)

test = (
    [10, 12, 11, 0, 0, 1],
    [12, 12, 0, 0, 0, 1],
    [7, 0, 0, 10, 6, 11],
)

error = []
v_error = []
print("Training starts...")
prev = time.time()
for i in range(1200):
    r = random.randint(0, len(training) - 1)
    ensemble.fit(*training[r])
    error.append(ensemble.error)

    if i % 10 == 0:
        v_error.append(cv(ensemble, validation))
print(f"Training is finished! Spend time: {time.time() - prev:.2f}")

for t in test:
    print(f"{t} -> {normal(ensemble.get(t)) * 50}")

plot.title("Learning error")
plot.plot(error)
plot.plot([i * 10 for i in range(len(v_error))], v_error)
plot.show()