import numpy

from core import FeedForward


def cv(nn: FeedForward, data) -> float:
    """
    Cross Validation

    Calculate testing error of neural network

    :param nn:
    :param data:
    :return:
    """
    error = 0.0
    for v in data:
        x, d = v[0], v[1]
        error += numpy.abs(d - nn.get(x)).sum()
    return float(error / len(data))
