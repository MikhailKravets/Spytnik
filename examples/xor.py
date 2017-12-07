import matplotlib.pyplot as plot
import random

import layers
from core import FeedForward


nn = FeedForward(momentum=0.1, learn_rate=0.1, weight_decay=0.2)  # .create([2, 2, 1], default=layers.Tanh)

nn += layers.Input(2, 2)
nn += layers.Tanh(2, 2)
nn += layers.Linear(2, 1)

s = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]

E = []

for i in range(10_000):
    r = random.randint(0, len(s) - 1)
    nn.fit(*s[r])
    E.append(nn.error)

print(nn)

for v in s:
    print(nn.get(v[0]), end='\n\n')

plot.title("Training error")
plot.plot(E, label='Training error')
plot.show()