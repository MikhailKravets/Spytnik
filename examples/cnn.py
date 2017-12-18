import struct
import numpy as np
import matplotlib.pyplot as pl


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def conv2D(tensor: np.ndarray):
    kernel = np.random.random(size=(5, 5))
    sr, sc = kernel.shape[0], kernel.shape[1]

    shape = tensor.shape[0] - kernel.shape[0] + 1, tensor.shape[1] - kernel.shape[1] + 1
    convulted = np.zeros(shape=shape)

    for i in range(convulted.shape[0]):
        for j in range(convulted.shape[1]):
            convulted[i, j] = np.sum(tensor[i:i + sr, j:j + sc] * kernel)
    print(convulted)
    return convulted


first = read_idx('train-images.idx')[0] / 256

tensor = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [1, 1, -1, -1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 0, -1, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, -1, 0],
    [1, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, -1, 0, 0, 1],
    [1, -1, 0, 0, 1, 0, 1],
])


pl.imshow(conv2D(first), cmap='Greys_r')
pl.show()