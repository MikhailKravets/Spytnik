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

noise(training, from_range=(0, 3), axis=0)
noise(training, from_range=(0, 0.1), axis=1)

nn = FeedForward(learn_rate=0.05, momentum=0.00, weight_decay=0.6)

nn += layers.Linear(6, 110)
nn += layers.Tanh(110, 250)
nn += layers.Tanh(250, 275)
nn += layers.Tanh(275, 275)
nn += layers.Tanh(275, 275)
nn += layers.Tanh(275, 275)
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