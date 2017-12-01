"""
This is the hard one. I need to handle it by my own
"""
import random

import matplotlib.pyplot as plot
import time

import layers
from core import FeedForward, from_csv, separate_data, noise
from core.estimators import cv


training, validation = separate_data(from_csv("D:\\DELETE\\Дипломмо\\output.csv"), 0.15)

noise(training, from_range=(0, 2), axis=0)
noise(training, from_range=(-0.05, 0.05), axis=1)

nn = FeedForward(learn_rate=0.05, momentum=0.2, weight_decay=0.5)

nn += layers.Linear(6, 260)
nn += layers.Tanh(260, 456)
nn += layers.Relu(456, 456)
nn += layers.Tanh(456, 456)
nn += layers.Relu(456, 456)
nn += layers.Tanh(456, 456)
nn += layers.Relu(456, 456)
nn += layers.Relu(456, 275)
nn += layers.Tanh(275, 50)
nn += layers.Tanh(50, 8)
nn += layers.Linear(8, 0)

error = []
v_error = []
print("Training starts...")
prev = time.time()
for i in range(900):
    r = random.randint(0, len(training) - 1)
    nn.fit(*training[r])
    error.append(nn.error)

    if i % 10 == 0:
        v_error.append(cv(nn, validation))
print(f"Training is finished! Spend time: {time.time() - prev:.2f}")

plot.title("Learning error")
plot.plot(error)
plot.plot([i * 10 for i in range(len(v_error))], v_error)
plot.show()