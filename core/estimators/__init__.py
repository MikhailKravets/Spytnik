import numpy

from core import FeedForward


def cv(nn: FeedForward, x, d) -> float:
    """
    Cross Validation

    Calculate testing error of neural network

    :param nn:
    :param x:
    :param d:
    :return:
    """
    error = 0
    for xv, dv in zip(x, d):
        error += numpy.abs(d - nn.get(x)).sum()
    return error
