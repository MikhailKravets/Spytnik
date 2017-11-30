"""
This is the hard one. I need to handle it by my own
"""
import random

import matplotlib.pyplot as plot
import time

import layers
from core import FeedForward, from_csv, separate_data
from core.estimators import cv


training, validation = separate_data(from_csv("D:\\DELETE\\Дипломмо\\output.csv"), 0.15)

nn = FeedForward(learn_rate=0.05, momentum=0.2, weight_decay=0.7)

nn += layers.Linear(6, 110)
nn += layers.Tanh(110, 150)
nn += layers.Tanh(150, 175)
nn += layers.Relu(175, 175)
nn += layers.Relu(175, 175)
nn += layers.Relu(175, 175)
nn += layers.Tanh(175, 50)
nn += layers.Tanh(50, 8)
nn += layers.Linear(8, 0)

error = []
v_error = []
print("Training starts...")
prev = time.time()
for i in range(20_000):
    r = random.randint(0, len(training) - 1)
    nn.fit(*training[r])
    error.append(nn.error)

    if i % 300 == 0:
        v_error.append(cv(nn, validation))
    if i == 7_000:
        print("And it is still training...")
    elif i == 13_000:
        print("Almost half...")
    if i == 17_000:
        print("I see the end...")
print(f"Training is finished! Spend time: {time.time() - prev:.2f}")

plot.title("Learning error")
plot.plot(error)
plot.plot(v_error)
plot.show()