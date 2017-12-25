import struct
import numpy as np
import matplotlib.pyplot as pl


def im2col(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]


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

t2 = np.array([
    [1, 3, 5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14, 15, 16],
    [17, 18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36, 37],
    [38, 39, 40, 41, 42, 43, 44],
    [45, 46, 47, 48, 49, 50, 51],
])

print(first.shape)
coled_im = im2col(first, (5, 5))
print(coled_im.shape)

kernel = np.random.random(size=(5, 5))
coled_kern = im2col(kernel, (5, 5))
print(coled_kern.shape)

convo = coled_kern.T.dot(coled_im).reshape(24, 24)
print(convo.shape)

pl.imshow(convo, cmap='Greys_r')
pl.show()