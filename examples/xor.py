import matplotlib.pyplot as plot
import layers

from nn import FeedForward


nn = FeedForward() # .create([2, 2, 1], default=layers.Tanh)

nn += layers.Linear(2, 2)
nn += layers.Tanh(2, 1)
nn += layers.Linear(1, 0)

s = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]

E = []

for i in range(10_000):
    nn.fit(*s[i % 4])
    E.append(nn.error)

print(nn)

for v in s:
    print(nn.get(v[0]), end='\n\n')

plot.plot(E)
plot.show()